import numpy as np
import pyarrow as pa
import vector
import matplotlib.pyplot as plt


class SystematicsApplier:
    FEAT_PT = 1
    FEAT_ETA = 2
    FEAT_PHI = 3
    FEAT_E = 0

    # feature names for clarity (shared class-level)
    X_FEATURES = ["E", "pt", "eta", "phi", "btag", "isLepton", "charge"]
    COND_FEATURES = ["met", "met_phi", "nLepton", "nbJet", "nJet", "HT", "HT_lep", "M_all", "M_leps", "M_bjets"]

    # normalization rules declared ONCE
    X_NORM = {
        "E": "log_normalize",
        "pt": "log_normalize",
        "eta": "none",
        "phi": "none",
        "btag": "none",
        "isLepton": "none",
        "charge": "none",
    }

    COND_NORM = {
        "met": "log_normalize",
        "met_phi": "normalize",
        "nLepton": "none",
        "nbJet": "none",
        "nJet": "none",
        "HT": "log_normalize",
        "HT_lep": "log_normalize",
        "M_all": "log_normalize",
        "M_leps": "log_normalize",
        "M_bjets": "log_normalize",
    }

    # ===============================================================
    # Normalization ↔ denormalization utils
    # ===============================================================
    @staticmethod
    def _log_norm_inverse(v):  # stored -> raw
        return np.expm1(v)  # inverse of log1p-normalization

    @staticmethod
    def _log_norm_forward(v):  # raw -> stored
        return np.log1p(v)

    @staticmethod
    def _norm_inverse(v):  # assume mean0-std1, user can replace later
        return v  # placeholder for real scaler

    @staticmethod
    def _norm_forward(v):
        return v

    def __init__(self, n_seq=18, n_feat=7, n_global=10):
        self.n_seq = n_seq
        self.n_feat = n_feat
        self.n_global = n_global

    # ===================================================================
    def apply(self, table, object_shifts=None, met_shift=None, recompute_globals=True):

        x, mask, cond = self._table_to_arrays(table)

        if recompute_globals:
            x, cond = self._recompute_energy_and_globals(x, mask, cond, object_shifts, met_shift)

        return self._arrays_to_table(table, x, cond)

    def _apply_object_shifts(self, x, mask, cfg):

        for name, rule in cfg.items():
            feats = rule["features"]  # ["pt","eta","phi"]
            select = rule["select"](x) & mask  # boolean mask
            apply = rule["apply"]
            scale = rule["scale"]

            for f in feats:
                idx = {"pt": 1, "eta": 2, "phi": 3}[f]
                before = x[..., idx].copy()
                x[..., idx] = np.where(select, apply(before, scale), before)

        return x

    # ===================================================================
    def _recompute_energy_and_globals(self, x, mask, cond, object_shifts=None, met_shift=None):

        # =======================================================
        # 1. Extract original state BEFORE shift (needed for mass)
        # =======================================================
        pt0 = x[..., self.FEAT_PT]
        eta0 = x[..., self.FEAT_ETA]
        phi0 = x[..., self.FEAT_PHI]
        E0 = x[..., self.FEAT_E]  # old energy used to recover mass

        px0 = pt0 * np.cos(phi0)
        py0 = pt0 * np.sin(phi0)
        pz0 = pt0 * np.sinh(eta0)
        mass = np.sqrt(np.maximum(E0 * E0 - (px0 * px0 + py0 * py0 + pz0 * pz0), 0))

        # =======================================================
        # 2. Apply object-level shifts HERE (in real space)
        # =======================================================
        if object_shifts:
            for name, rule in object_shifts.items():
                sel = rule["select"](x) & mask
                apply = rule["apply"]
                s = rule["scale"]

                if "pt" in rule["features"]:
                    pt0 = np.where(sel, apply(pt0, s), pt0)

                if "eta" in rule["features"]:
                    eta0 = np.where(sel, apply(eta0, s), eta0)

                if "phi" in rule["features"]:
                    phi0 = np.where(sel, apply(phi0, s), phi0)

        # =======================================================
        # 3. Recompute momenta & new energy with preserved mass
        # =======================================================
        px = pt0 * np.cos(phi0)
        py = pt0 * np.sin(phi0)
        pz = pt0 * np.sinh(eta0)
        E = np.sqrt(mass * mass + px * px + py * py + pz * pz)
        x[..., self.FEAT_PT] = pt0
        x[..., self.FEAT_ETA] = eta0
        x[..., self.FEAT_PHI] = phi0
        x[..., self.FEAT_E] = E

        vec = vector.array({"px": px, "py": py, "pz": pz, "E": E})

        # =======================================================
        # 4. Compute global objects & masses
        # =======================================================
        lep = x[..., 5]
        btag = x[..., 4]
        mlep = (mask & (lep != 0))
        mb = (mask & (btag == 1))
        mjet = (mask & (lep == 0) & (btag == 0))

        # Build group vectors using masked pt
        vec_lep = vector.array(
            {"px": px * (lep != 0), "py": py * (lep != 0), "pz": pz * (lep != 0), "E": E * (lep != 0)})
        vec_bjet = vector.array(
            {"px": px * (btag == 1), "py": py * (btag == 1), "pz": pz * (btag == 1), "E": E * (btag == 1)})
        vec_jet = vector.array(
            {
                "px": px * ((lep == 0) & (btag == 0)), "py": py * ((lep == 0) & (btag == 0)),
                "pz": pz * ((lep == 0) & (btag == 0)), "E": E * ((lep == 0) & (btag == 0))
            })

        cond[:, 2] = mlep.sum(1)
        cond[:, 3] = mb.sum(1)
        cond[:, 4] = mjet.sum(1)
        cond[:, 5] = (pt0 * mjet).sum(1) + (pt0 * mlep).sum(1)
        cond[:, 6] = (pt0 * mlep).sum(1)

        cond[:, 7] = vec.sum(1).mass
        cond[:, 8] = vec_lep.sum(1).mass
        cond[:, 9] = vec_bjet.sum(1).mass

        # =======================================================
        # 5. MET shift fully inside this function
        # =======================================================
        vtot = vec.sum(1)
        MET = vtot.pt
        phiMET = vtot.phi

        if met_shift:
            dpx = met_shift.get("px", 0)
            dpy = met_shift.get("py", 0)
            pxm = MET * np.cos(phiMET) + dpx
            pym = MET * np.sin(phiMET) + dpy
            MET = np.sqrt(pxm * pxm + pym * pym)
            phiMET = np.arctan2(pym, pxm)

        cond[:, 0] = MET
        cond[:, 1] = (phiMET + np.pi) % (2 * np.pi)

        return x, cond

    # ===================================================================
    def _apply_met_shift(self, cond, dpx, dpy):
        MET = cond[:, 0]
        phi = cond[:, 1]
        px = MET * np.cos(phi) + dpx
        py = MET * np.sin(phi) + dpy
        cond[:, 0] = np.sqrt(px * px + py * py)
        cond[:, 1] = np.arctan2(py, px)
        return cond

    # ---- parquet I/O (unchanged) ----
    # ===================================================================
    def _table_to_arrays(self, table):
        N = table.num_rows
        x = np.zeros((N, self.n_seq, self.n_feat), np.float32)
        m = np.ones((N, self.n_seq), bool)
        c = np.zeros((N, self.n_global), np.float32)

        for s in range(self.n_seq):
            for f in range(self.n_feat):
                col = f"x:{s}:{f}"
                if col in table.column_names:
                    raw = table[col].to_numpy()
                    key = self.X_FEATURES[f]
                    if self.X_NORM[key] == "log_normalize":
                        x[:, s, f] = self._log_norm_inverse(raw)
                    elif self.X_NORM[key] == "normalize":
                        x[:, s, f] = self._norm_inverse(raw)
                    else:
                        x[:, s, f] = raw

        for s in range(self.n_seq):
            col = f"x_mask:{s}"
            if col in table.column_names:
                m[:, s] = table[col].to_numpy()

        for k in range(self.n_global):
            col = f"conditions:{k}"
            if col in table.column_names:
                raw = c[:, k] = table[col].to_numpy()
                key = self.COND_FEATURES[k]
                if self.COND_NORM[key] == "log_normalize":
                    c[:, k] = self._log_norm_inverse(raw)
                elif self.COND_NORM[key] == "normalize":
                    c[:, k] = self._norm_inverse(raw)
                else:
                    c[:, k] = raw

        return x, m, c

    def _arrays_to_table(self, table, x, cond):
        arrs = {n: table[n] for n in table.column_names}

        for s in range(self.n_seq):
            for f in range(self.n_feat):
                col = f"x:{s}:{f}"
                if col in arrs:
                    key = self.X_FEATURES[f]
                    v = x[:, s, f]
                    if self.X_NORM[key] == "log_normalize":
                        v = self._log_norm_forward(v)
                    elif self.X_NORM[key] == "normalize":
                        v = self._norm_forward(v)
                    arrs[col] = pa.array(v)

        for k in range(self.n_global):
            col = f"conditions:{k}"
            if col in arrs:
                key = self.COND_FEATURES[k]
                v = cond[:, k]
                if self.COND_NORM[key] == "log_normalize":
                    v = self._log_norm_forward(v)
                elif self.COND_NORM[key] == "normalize":
                    v = self._norm_forward(v)
                arrs[col] = pa.array(v)

        return pa.Table.from_arrays(
            [arrs[n] for n in table.column_names], names=list(table.column_names)
        )

    def plot_feature_shift(
            self,
            table_before,
            table_after,
            feature="pt",  # "pt","eta","phi","E" or condition name
            selection=lambda x: np.ones_like(x[..., 1], bool),
            bins=60,
            range=None,
            logy=False,
            title=None,
            save=None
    ):
        """
        Plot histogram of a feature before and after systematics.

        feature options:
        ----------------
        Object-level (x):
            "pt", "eta", "phi", "E"

        Global conditions (cond):
            "met","met_phi","nLepton","nbJet","nJet",
            "HT","HT_lep","M_all","M_leps","M_bjets"
            or "cond:0" .. "cond:9"
        """

        # mapping for object features
        feat_map = {"E": 0, "pt": 1, "eta": 2, "phi": 3}

        # mapping for condition columns
        cond_map = {
            "met": 0, "met_phi": 1, "nLepton": 2, "nbJet": 3, "nJet": 4,
            "HT": 5, "HT_lep": 6, "M_all": 7, "M_leps": 8, "M_bjets": 9
        }

        # ---------------------------------------------------
        # Decide which feature we are plotting
        # ---------------------------------------------------
        is_condition = False

        if feature in feat_map:
            feat_type = "x"
            feat_idx = feat_map[feature]

        elif feature in cond_map:
            feat_type = "cond"
            feat_idx = cond_map[feature]
            is_condition = True

        elif feature.startswith("cond:"):  # e.g. "cond:3"
            feat_type = "cond"
            feat_idx = int(feature.split(":")[1])
            is_condition = True

        else:
            raise ValueError(f"Unknown feature '{feature}'. Valid options: "
                             f"{list(feat_map.keys()) + list(cond_map.keys())}")

        # ---------------------------------------------------
        # Load data
        # ---------------------------------------------------
        x0, mask0, c0 = self._table_to_arrays(table_before)
        x1, mask1, c1 = self._table_to_arrays(table_after)

        # ---------------------------------------------------
        # Extract values
        # ---------------------------------------------------
        if feat_type == "x":
            sel0 = selection(x0) & mask0
            sel1 = selection(x1) & mask1
            vals0 = x0[..., feat_idx][sel0].flatten()
            vals1 = x1[..., feat_idx][sel1].flatten()

        else:  # condition feature
            vals0 = c0[:, feat_idx].flatten()
            vals1 = c1[:, feat_idx].flatten()

        # ---------------------------------------------------
        # Plot
        # ---------------------------------------------------
        def _pretty_hist_comparison(vals0, vals1, feature, bins, range, logy, title, save=None):

            # style
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(7, 5), dpi=140)

            # colors
            c_before = "#2b78e4"  # royal blue
            c_after = "#d62728"  # ATLAS red

            # histogram
            hist0 = ax.hist(vals0, bins=bins, range=range, density=True,
                            histtype="step", linewidth=2, color=c_before, label="Before")
            ax.hist(vals0, bins=bins, range=range, density=True,
                    alpha=0.20, color=c_before)

            hist1 = ax.hist(vals1, bins=bins, range=range, density=True,
                            histtype="step", linewidth=2, color=c_after, label="After")
            ax.hist(vals1, bins=bins, range=range, density=True,
                    alpha=0.20, color=c_after)

            # labels
            ax.set_xlabel(feature, fontsize=13)
            ax.set_ylabel("Normalized entries", fontsize=13)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.30)

            if logy: ax.set_yscale("log")
            ax.set_title(title or f"Systematic variation: {feature}", fontsize=14, pad=10)

            # Make axis nicer
            ax.tick_params(axis='both', labelsize=11, length=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if save:
                fig.savefig(save, bbox_inches="tight", dpi=200)
                print(f"Saved → {save}")

            plt.show()

        _pretty_hist_comparison(vals0, vals1, feature, bins, range, logy, title, save)


if __name__ == '__main__':
    object_shifts = {
        "jet_pt_up": {
            "features": ["pt"],  # only shift pT (column index 1)
            "select": lambda x: (x[..., 5] == 0),  # jets = not lepton
            "apply": lambda v, s: v * (1 + s),  # multiplicative scale
            "scale": 0.05  # +5%
        },
        # "jet_pt_down": {
        #     "features": ["pt"],  # only shift pT (column index 1)
        #     "select": lambda x: (x[..., 5] == 0),  # jets = not lepton
        #     "apply": lambda v, s: v * (1 + s),  # multiplicative scale
        #     "scale": -0.05  # +5%
        # }
    }

    import pyarrow.parquet as pq

    base = pq.read_table(
        "/Users/avencastmini/PycharmProjects/EveNet/downstreams/HHML/train_input/k_fold/fold_0/test/train.parquet")

    syst = SystematicsApplier()

    # shift jet pT → recompute energy + globals (MET, HT, masses...)
    shifted = syst.apply(
        base,
        object_shifts=object_shifts,
        met_shift=None,  # no MET smearing
        recompute_globals=True  # mandatory for physics consistency
    )

    syst.plot_feature_shift(
        base, shifted,
        feature="pt",
        selection=lambda x: (x[..., 5] == 0),  # only jets
        bins=50,
        logy=False,
        range=[0, 300],
        # save="jet_pt_shift.png"
    )

    syst.plot_feature_shift(
        base, shifted,
        feature="pt",
        selection=lambda x: (x[..., 5] != 0),  # only jets
        bins=50,
        logy=False,
        range=[0, 300],
        # save="jet_pt_shift.png"
    )

    syst.plot_feature_shift(
        base, shifted,
        feature="HT",
        bins=50,
        logy=False,
        range=[0, 600],
        # save="jet_pt_shift.png"
    )

    syst.plot_feature_shift(
        base, shifted,
        feature="met",
        bins=50,
        logy=False,
        range=[0, 600],
        # save="jet_pt_shift.png"
    )

    syst.plot_feature_shift(
        base, shifted,
        feature="M_leps",
        bins=50,
        logy=False,
        range=[0, 300],
        # save="jet_pt_shift.png"
    )
    # pq.write_table(shifted, "jets_ptUp.parquet")
    print(">> jet pT shifted +5% & saved to jets_ptUp.parquet")
