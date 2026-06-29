# =========================================================
# IMPORTS
# =========================================================

import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata

from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel
)

from sklearn.ensemble import RandomForestRegressor

import shap

warnings.filterwarnings("ignore")


# =========================================================
# PLOT STYLE
# =========================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 2,
})


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = "../data/processed/charts"

MODEL_DIR = "../data/processed/models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# DATA LOADING
# =========================================================

def load_data(path):
    return pd.read_csv(path)


# =========================================================
# YIELD PROCESSING
# =========================================================

YIELD_COLS = [
    "BINOL Yield (%) 1",
    "BINOL Yield (%) 2",
    "BINOL Yield (%) 3"
]


def aggregate_yields(df):

    df["yield_mean"] = df[YIELD_COLS].mean(
        axis=1,
        skipna=True
    )

    df["yield_std"] = df[YIELD_COLS].std(
        axis=1,
        skipna=True
    )

    df["yield_cv"] = (
        df["yield_std"] /
        (df["yield_mean"] + 1e-8)
    )

    df["n_rep"] = df[YIELD_COLS].count(axis=1)

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def engineer_features(df):

    df["mix_ratio_num"] = df["mix ratio"]

    df["chelating_ratio_num"] = (
        df["chelating agent ratio"]
    )

    # -----------------------------------------------------
    # LOG-STABILIZED Fe/W RATIO
    # -----------------------------------------------------

    df["Fe_W_ratio"] = np.log1p(
        df["Fe wt.%"] /
        (df["W wt.%"] + 1e-3)
    )

    # -----------------------------------------------------
    # INTERACTION FEATURES
    # -----------------------------------------------------

    df["temp_Fe"] = (
        df["Calcination Temperature [ºC]"] *
        df["Fe wt.%"]
    )

    df["temp_W"] = (
        df["Calcination Temperature [ºC]"] *
        df["W wt.%"]
    )

    # -----------------------------------------------------
    # POLYNOMIAL FEATURES
    # -----------------------------------------------------

    df["temp_sq"] = (
        df["Calcination Temperature [ºC]"] ** 2
    )

    df["Fe_sq"] = df["Fe wt.%"] ** 2

    df["W_sq"] = df["W wt.%"] ** 2

    return df


# =========================================================
# OUTLIER STABILIZATION
# =========================================================

def stabilize_outliers(df):

    # -----------------------------------------------------
    # REMOVE EXTREME Fe/W RATIO OUTLIERS
    # -----------------------------------------------------

    upper = df["Fe_W_ratio"].quantile(0.99)

    df = df[
        df["Fe_W_ratio"] <= upper
    ].copy()

    # -----------------------------------------------------
    # REMOVE IMPOSSIBLE YIELDS
    # -----------------------------------------------------

    df = df[
        (df["yield_mean"] >= 0) &
        (df["yield_mean"] <= 100)
    ]

    return df


# =========================================================
# VISUALIZATION
# =========================================================

def plot_correlation(df):

    cols = [
        "Calcination Temperature [ºC]",
        "Fe_W_ratio",
        "mix_ratio_num",
        "chelating_ratio_num",
        "yield_mean",
        "yield_std",
        "yield_cv"
    ]

    corr = df[cols].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True
    )

    plt.title("Feature Correlation")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/correlation_heatmap.pdf"
    )

    plt.close()


def plot_lowess_temp_vs_yield(df):

    x = df["Calcination Temperature [ºC]"]

    y = df["yield_mean"]

    lowess_result = lowess(
        endog=y,
        exog=x,
        frac=0.35
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        x,
        y,
        alpha=0.7,
        label="Experiments"
    )

    plt.plot(
        lowess_result[:, 0],
        lowess_result[:, 1],
        linewidth=3,
        label="LOWESS Trend"
    )

    plt.xlabel("Calcination Temperature (ºC)")

    plt.ylabel("Average Yield (%)")

    plt.title(
        "LOWESS:\nTemperature vs BINOL Yield"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/lowess_temp_vs_yield.pdf"
    )

    plt.close()


# =========================================================
# GPR MODELING
# =========================================================

FEATURES = [
    "Calcination Temperature [ºC]",
    "Fe_W_ratio",
    "mix_ratio_num",
    "chelating_ratio_num",
    "temp_sq"
]


def prepare_data(df):

    clean_df = df[
        FEATURES + ["yield_mean"]
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    X = clean_df[FEATURES]

    y = clean_df["yield_mean"]

    return X, y


def train_gpr_model(X, y):

    # -----------------------------------------------------
    # STANDARDIZATION
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------------------------
    # GPR KERNEL
    # -----------------------------------------------------

    kernel = (
        ConstantKernel(1.0)
        *
        RBF(length_scale=1.0)
        +
        WhiteKernel(noise_level=1)
    )

    # -----------------------------------------------------
    # GPR MODEL
    # -----------------------------------------------------

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=20,
        random_state=42
    )

    gpr.fit(X_train, y_train)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    pred_mean, pred_std = gpr.predict(
        X_test,
        return_std=True
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        pred_mean
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred_mean
        )
    )

    r2 = r2_score(
        y_test,
        pred_mean
    )

    print("\n==============================")
    print("GPR PERFORMANCE")
    print("==============================")

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    # -----------------------------------------------------
    # UNCERTAINTY PLOT
    # -----------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        y_test,
        pred_mean,
        yerr=pred_std,
        fmt='o',
        alpha=0.7,
        label="Predictions"
    )

    # -----------------------------------------------------
    # IDEAL y = x REFERENCE LINE
    # -----------------------------------------------------

    line_min = min(
        np.min(y_test),
        np.min(pred_mean)
    )

    line_max = max(
        np.max(y_test),
        np.max(pred_mean)
    )

    plt.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=2,
        color="red",
        label="Ideal Prediction (y=x)"
    )

    plt.xlabel("True Yield (%)")

    plt.ylabel("Predicted Yield (%)")

    plt.title(
        "GPR Predictions with Uncertainty"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/gpr_uncertainty.pdf"
    )

    plt.close()

    return gpr, scaler, X_scaled


# =========================================================
# SHAP INTERPRETATION
# =========================================================

def shap_analysis(X, y):

    # -----------------------------------------------------
    # RANDOM FOREST FOR SHAP
    # -----------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X, y)

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    explainer = shap.Explainer(model)

    shap_values = explainer(X)

    # -----------------------------------------------------
    # SUMMARY PLOT
    # -----------------------------------------------------

    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/shap_summary.pdf"
    )

    plt.close()

    # -----------------------------------------------------
    # BAR PLOT
    # -----------------------------------------------------

    shap.plots.bar(
        shap_values,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/shap_bar.pdf"
    )

    plt.close()

    print("\nSHAP analysis completed.")


# =========================================================
# BAYESIAN OPTIMIZATION STYLE RECOMMENDATION
# =========================================================

def recommend_experiments(
    gpr,
    scaler
):

    # -----------------------------------------------------
    # GENERATE CANDIDATES
    # -----------------------------------------------------

    n_candidates = 5000

    candidate_df = pd.DataFrame({

        "Calcination Temperature [ºC]":
        np.random.uniform(
            300,
            800,
            n_candidates
        ),

        "Fe_W_ratio":
        np.random.uniform(
            df["Fe_W_ratio"].min(),
            df["Fe_W_ratio"].max(),
            n_candidates
        ),

        "mix_ratio_num":
        np.random.uniform(
            0,
            10,
            n_candidates
        ),

        "chelating_ratio_num":
        np.random.uniform(
            0,
            10,
            n_candidates
        )
    })

    # polynomial feature
    candidate_df["temp_sq"] = (
        candidate_df[
            "Calcination Temperature [ºC]"
        ] ** 2
    )

    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    X_candidate = scaler.transform(
        candidate_df[FEATURES]
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    pred_mean, pred_std = gpr.predict(
        X_candidate,
        return_std=True
    )

    # -----------------------------------------------------
    # UCB ACQUISITION
    # -----------------------------------------------------

    candidate_df["predicted_yield"] = pred_mean

    candidate_df["uncertainty"] = pred_std

    candidate_df["score"] = (
        pred_mean +
        1.96 * pred_std
    )

    # -----------------------------------------------------
    # TOP CANDIDATES
    # -----------------------------------------------------

    recommendations = candidate_df.sort_values(
        by="score",
        ascending=False
    ).head(30)

    print("\n==============================")
    print("TOP RECOMMENDED EXPERIMENTS")
    print("==============================")

    print(
        recommendations[
            [
                "Calcination Temperature [ºC]",
                "Fe_W_ratio",
                "predicted_yield",
                "uncertainty",
                "score"
            ]
        ]
    )

    recommendations.to_csv(
        "../data/processed/recommended_experiments.csv",
        index=False
    )

    return recommendations


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    df = load_data(
        "../data/raw/Fe-W Catalyst-sobol sequence data 2.csv"
    )

    # -----------------------------------------------------
    # PROCESSING
    # -----------------------------------------------------

    df = aggregate_yields(df)

    df = engineer_features(df)

    df = stabilize_outliers(df)

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    plot_correlation(df)

    plot_lowess_temp_vs_yield(df)

    # -----------------------------------------------------
    # MODEL DATA
    # -----------------------------------------------------

    X, y = prepare_data(df)

    # -----------------------------------------------------
    # GPR
    # -----------------------------------------------------

    gpr, scaler, X_scaled = train_gpr_model(
        X,
        y
    )

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    shap_analysis(X, y)

    # -----------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------

    recommend_experiments(
        gpr,
        scaler
    )

    print("\nPipeline completed.")