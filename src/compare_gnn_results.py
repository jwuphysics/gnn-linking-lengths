from pathlib import Path
import gc
import numpy as np
import pandas as pd
import scipy
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).parent.parent.resolve()

verbose = True


def xgboost_cv(df, feature_cols, target_col, kfold_col, nfold=3, **xgb_kwargs):
    """Cross valiate using xgboost model"""

    xgb_preds = []
    for k in range(nfold):
        valid_mask = (df[kfold_col] == k)
        
        X_train = df[feature_cols][~valid_mask]
        y_train = df[target_col][~valid_mask]
        X_valid = df[feature_cols][valid_mask]
        y_valid = df[target_col][valid_mask]

        xgb = XGBRegressor(
            tree_method="auto",
            objective="reg:squarederror",
            random_state=42,
            **xgb_kwargs
        )
        xgb.fit(X_train, y_train)

        xgb_preds.append(xgb.predict(X_valid))

        if ("verbosity" in xgb_kwargs) and (xgb_kwargs["verbosity"] > 0):
            print(mean_squared_error(xgb_preds[-1], y_valid)**0.5)
    xgb_preds = np.concatenate(xgb_preds)
    return xgb_preds



def collate_comparison_results(D_link):
    """Runs XGBoost and SHAM models and appends results into final cross-validated table."""

    # get all DMO and Hydro results
    results_path = f"{ROOT}/results/linking_length_tests/D_link{D_link}"
    dmo = []
    hydro = []
    kfold = []
    for k in range(3):
        dmo += [pd.read_csv(f"{results_path}/gnn-1000-multi-dmo-fold_{k+1}.cat", dtype={"is_central": bool})]
        hydro += [pd.read_csv(f"{results_path}/gnn-1000-multi-hydro-fold_{k+1}.cat")]
        kfold += [np.full(len(dmo[-1]), k)]

    dmo = pd.concat(dmo).set_index("subhalo_id", drop=True)
    hydro = pd.concat(hydro).set_index("subhalo_id", drop=True)

    df = dmo.reset_index().join(hydro[["p_GNN_hydro"]], on="subhalo_id").set_index("subhalo_id")
    df["k_fold"] = np.concatenate(kfold)

    # get features ready for tabular models
    halo_dmo_features = ['log_Mhalo_dmo', 'log_Vmax_dmo', 'is_central']
    halo_hydro_features = ['log_Mhalo_hydro', 'log_Vmax_hydro', 'is_central']
    disperse_features = ['d_minima', 'd_node', 'd_saddle_1', 'd_saddle_2', 'd_skel']
    overdensity_features = ['overdensity']
    target = ['log_Mstar']

    # run xgboost models
    xgb_Mstar_dmo = xgboost_cv(df, halo_dmo_features, target, "k_fold", nfold=3, verbosity=0).astype(float)
    xgb_Mstar_overdensity_dmo = xgboost_cv(df, halo_dmo_features + overdensity_features, target, "k_fold", nfold=3, verbosity=0).astype(float)
    xgb_Mstar_disperse_dmo = xgboost_cv(df, halo_dmo_features + disperse_features, target, "k_fold", nfold=3, verbosity=0).astype(float)
    xgb_Mstar_hydro = xgboost_cv(df, halo_hydro_features, target, "k_fold", nfold=3, verbosity=0).astype(float)
    xgb_Mstar_overdensity_hydro = xgboost_cv(df, halo_hydro_features + overdensity_features, target, "k_fold", nfold=3, verbosity=0).astype(float)
    xgb_Mstar_disperse_hydro = xgboost_cv(df, halo_hydro_features + disperse_features, target, "k_fold", nfold=3, verbosity=0).astype(float)

    # Vmax - SHAM models
    sham_dmo_central = np.vstack([list(reversed(sorted((df.log_Vmax_dmo[df.is_central])))), list(reversed(sorted(df.log_Mstar[df.is_central])))])
    sham_dmo_satellite = np.vstack([list(reversed(sorted((df.log_Vmax_dmo[~df.is_central])))), list(reversed(sorted(df.log_Mstar[~df.is_central])))])
    sham_dmo_central_interp = lambda Mh: np.interp(Mh, *sham_dmo_central.T[::-1].T)
    sham_dmo_satellite_interp = lambda Mh: np.interp(Mh, *sham_dmo_satellite.T[::-1].T)
    sham_dmo_predictions = lambda Mh, cen: np.where(cen, sham_dmo_central_interp(Mh), sham_dmo_satellite_interp(Mh))
    Mstar_sham_dmo = np.array([sham_dmo_predictions(Mh, cen) for Mh, cen in zip(df.log_Vmax_dmo, df.is_central)], dtype=float)

    sham_hydro_central = np.vstack([list(reversed(sorted((df.log_Vmax_hydro[df.is_central])))), list(reversed(sorted(df.log_Mstar[df.is_central])))])
    sham_hydro_satellite = np.vstack([list(reversed(sorted((df.log_Vmax_hydro[~df.is_central])))), list(reversed(sorted(df.log_Mstar[~df.is_central])))])
    sham_hydro_central_interp = lambda Mh: np.interp(Mh, *sham_hydro_central.T[::-1].T)
    sham_hydro_satellite_interp = lambda Mh: np.interp(Mh, *sham_hydro_satellite.T[::-1].T)
    sham_hydro_predictions = lambda Mh, cen: np.where(cen, sham_hydro_central_interp(Mh), sham_hydro_satellite_interp(Mh))
    Mstar_sham_hydro = np.array([sham_hydro_predictions(Mh, cen) for Mh, cen in zip(df.log_Vmax_hydro, df.is_central)], dtype=float)

    df["p_xgb_dmo"] = xgb_Mstar_dmo
    df["p_xgb_overdensity_dmo"] = xgb_Mstar_overdensity_dmo
    df["p_xgb_disperse_dmo"] = xgb_Mstar_disperse_dmo
    df["p_xgb_hydro"] = xgb_Mstar_hydro
    df["p_xgb_overdensity_hydro"] = xgb_Mstar_overdensity_hydro
    df["p_xgb_disperse_hydro"] = xgb_Mstar_disperse_hydro
    df["p_sham_dmo"] = Mstar_sham_dmo
    df["p_sham_hydro"] = Mstar_sham_hydro

    return df





if __name__ == "__main__":
    for D_linkconda  in [0.3, 1, 3, 5, 10, 0.5, 1.5, 2, 2.5, 3.5, 4, 7.5,]: 
        gc.collect()

        cv_results_fname= f"{ROOT}/results/linking_length_tests/D_link{D_link}/cv-results.parquet"
        if not os.path.isfile(cv_results_fname):
            if verbose: print(f"Collating results for {D_link} Mpc")
            cv_results = collate_comparison_results(D_link)
            cv_results.to_parquet(cv_results_fname)