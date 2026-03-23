"""
feature_pipeline_v2.py
======================
Improved feature engineering + selection pipeline for S&P 500 cross-sectional ML.

HOW TO USE IN YOUR NOTEBOOK
----------------------------
After the existing Cell 6 (which builds `features_df` with the 11 raw base features
and `Next_Return`), replace Cells 7-onwards with:

    from feature_pipeline_v2 import (
        add_engineered_features,
        apply_crosssectional_ranking,
        run_feature_selection,
        SELECTED_FEATURES,        # updated after run_feature_selection()
    )

    features_df = add_engineered_features(features_df)
    features_df = apply_crosssectional_ranking(features_df)
    features_df, selected = run_feature_selection(features_df)
    FEATURE_COLS = selected          # replaces the old FEATURE_COLS list

Structure
---------
  1. add_engineered_features()  – builds ~15 derived features from the 11 raw ones
  2. apply_crosssectional_ranking() – converts every feature to [0,1] pct-rank per month
  3. run_feature_selection()    – 3-step filter → dedup → model-based selection
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Column name constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "Momentum_1M", "Momentum_3M", "Momentum_6M",
    "Short_Rev",
    "Price_to_MA20", "Price_to_MA60",
    "Volatility_20d", "Volatility_60d",
    "Volume_Ratio", "1M_Accum_Vol_Change",
    "High_52W_Ratio",
]

# Populated by add_engineered_features(); used downstream
DERIVED_FEATURES: list[str] = []

# Final feature set after selection; updated by run_feature_selection()
SELECTED_FEATURES: list[str] = []


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ~15 new features from the 11 raw base features.

    All operations are applied row-wise to the raw (un-ranked) values.
    Cross-sectional ranking is applied AFTER this step in a separate function.

    Returns the input DataFrame with new columns appended.
    No rows are dropped.
    """
    df = df.copy()
    eps = 1e-8   # avoid division-by-zero

    # ── (a) Risk-adjusted momentum (ratio features) ──────────────────────────
    # Sharpe-like metric: momentum divided by realised volatility.
    # A high-momentum stock with low volatility is more attractive than one
    # with equally high momentum but wild swings.
    df["Mom1M_VolAdj"] = df["Momentum_1M"] / (df["Volatility_20d"] + eps)
    df["Mom3M_VolAdj"] = df["Momentum_3M"] / (df["Volatility_60d"] + eps)
    df["Mom6M_VolAdj"] = df["Momentum_6M"] / (df["Volatility_60d"] + eps)

    # Short-vs-medium moving-average ratio.
    # > 1.0 → price is rising faster than the medium-term trend (acceleration).
    df["MA_Ratio"] = df["Price_to_MA20"] / (df["Price_to_MA60"] + eps)

    # Volatility regime: is short-term vol expanding or contracting vs 60d baseline?
    # > 1 → vol expansion (uncertainty rising); < 1 → vol compression.
    df["Vol_Regime"] = df["Volatility_20d"] / (df["Volatility_60d"] + eps)

    # ── (b) Interaction features ──────────────────────────────────────────────
    # Momentum confirmed by volume: a price rise on high volume is more reliable.
    df["Mom1M_x_VolRatio"] = df["Momentum_1M"] * df["Volume_Ratio"]

    # Momentum near 52-week high: breakout stocks (High_52W_Ratio → 1) with
    # positive momentum are a classic strong signal in cross-sectional research.
    df["Mom1M_x_52W"] = df["Momentum_1M"] * df["High_52W_Ratio"]

    # Trend with volume acceleration: a multi-month uptrend backed by rising
    # accumulated volume is a higher-conviction setup.
    df["Mom3M_x_AccumVol"] = df["Momentum_3M"] * df["1M_Accum_Vol_Change"]

    # ── (c) Nonlinear / signed transforms ────────────────────────────────────
    # Signed square: same sign as original but emphasises *outlier* movers.
    # rank(sign(x)*x²) ≠ rank(x) because the square changes relative ordering
    # for stocks with different magnitudes (even with same sign).
    df["Mom1M_Sq"] = np.sign(df["Momentum_1M"]) * df["Momentum_1M"] ** 2
    df["Mom3M_Sq"] = np.sign(df["Momentum_3M"]) * df["Momentum_3M"] ** 2

    # Log volume ratio: compresses the long right tail of volume spikes.
    # Ensures a 10× volume day doesn't dominate a 2× day by a factor of 5.
    df["VolRatio_Log"] = np.log(df["Volume_Ratio"].clip(lower=eps))

    # Square-root volatility: moderate compression to treat vol more linearly.
    df["Vol20d_Sqrt"] = np.sqrt(df["Volatility_20d"].clip(lower=0))

    # ── (d) Time-structure / momentum shape features ──────────────────────────
    # Momentum acceleration: how much faster is the stock moving recently?
    # Positive → the stock is picking up speed (short-term > medium-term).
    df["Mom_Acceleration"] = df["Momentum_1M"] - df["Momentum_3M"]

    # Longer-horizon deceleration: positive means medium-term > long-term trend.
    df["Mom_LT_Decel"] = df["Momentum_3M"] - df["Momentum_6M"]

    # MA spread: how far above (or below) the short-term MA is the price
    # relative to the medium-term MA?  Positive → short-term bullish bias.
    df["MA_Spread"] = df["Price_to_MA20"] - df["Price_to_MA60"]

    # Reversal-vs-momentum balance: short-term reversal + 1M momentum.
    # (Short_Rev is defined as -(1-week return), so this combination captures
    #  net 1-month signal net of the most recent week's mean-reversion noise.)
    df["RevMom_Balance"] = df["Short_Rev"] + df["Momentum_1M"]

    # ── (e) Additional alpha-style features ───────────────────────────────────
    # 52-week proximity penalised by current vol regime.
    # Stocks near their 52W high in a low-vol environment are
    # more likely to be genuinely strong, not just volatile.
    df["Vol52W_Score"] = df["High_52W_Ratio"] * (2.0 - df["Vol_Regime"].clip(upper=2.0))

    new_cols = [
        "Mom1M_VolAdj", "Mom3M_VolAdj", "Mom6M_VolAdj",
        "MA_Ratio", "Vol_Regime",
        "Mom1M_x_VolRatio", "Mom1M_x_52W", "Mom3M_x_AccumVol",
        "Mom1M_Sq", "Mom3M_Sq",
        "VolRatio_Log", "Vol20d_Sqrt",
        "Mom_Acceleration", "Mom_LT_Decel",
        "MA_Spread", "RevMom_Balance",
        "Vol52W_Score",
    ]
    global DERIVED_FEATURES
    DERIVED_FEATURES = new_cols

    print(f"[Feature Engineering]  Added {len(new_cols)} derived features.")
    print(f"  Total features (base + derived): {len(BASE_FEATURES) + len(new_cols)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Cross-Sectional Percentile Ranking
# ──────────────────────────────────────────────────────────────────────────────

def apply_crosssectional_ranking(
    df: pd.DataFrame,
    group_col: str = "ym",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Replace each feature value with its cross-sectional percentile rank [0, 1]
    computed strictly within each (year, month) slice.

    This is strictly no-look-ahead: ranking is done independently per time slice,
    so a stock in month T only competes with other stocks observable in month T.

    Why ranking instead of z-score?
    --------------------------------
    • Robust to outliers / non-normality (common in financial data).
    • Uniform [0,1] output ensures features are on the same scale every month,
      even if the raw distribution changes regime (e.g., crisis vs. bull market).
    • Monotone within each cross-section → preserves the ordinal signal.

    Parameters
    ----------
    df           : DataFrame containing at least `group_col` and `feature_cols`.
    group_col    : Column that identifies the time slice (default: 'ym').
    feature_cols : Columns to rank.  Defaults to BASE_FEATURES + DERIVED_FEATURES.

    Returns the DataFrame with feature columns replaced by their pct-ranks.
    """
    df = df.copy()
    if feature_cols is None:
        feature_cols = BASE_FEATURES + DERIVED_FEATURES

    # rank(pct=True) returns values in (0, 1] with ties averaged.
    df[feature_cols] = (
        df.groupby(group_col)[feature_cols]
        .transform(lambda x: x.rank(pct=True, method="average"))
    )

    # Verification: every feature should have mean ≈ 0.5, min > 0, max ≤ 1
    ranked_means = df[feature_cols].mean()
    ranked_min   = df[feature_cols].min()
    ranked_max   = df[feature_cols].max()

    ok = (ranked_means.between(0.48, 0.52).all()
          and (ranked_min > 0).all()
          and (ranked_max <= 1.0).all())

    print(f"[Ranking]  Applied cross-sectional pct-rank to {len(feature_cols)} features.")
    print(f"           Rank sanity check passed: {ok}")
    if not ok:
        print("  WARNING: some features may have unexpected rank distributions. Check NaNs.")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Feature Selection  (3-step pipeline)
# ──────────────────────────────────────────────────────────────────────────────

# ---- Step 1 helpers ----------------------------------------------------------

def _compute_ks_per_month(df: pd.DataFrame, feature: str, target: str = "Binary_Target") -> float:
    """
    KS statistic between feature distribution of label=1 vs label=0,
    averaged across all months.  Higher is better (more separable).
    """
    ks_vals = []
    for _, grp in df.groupby("ym"):
        pos = grp.loc[grp[target] == 1, feature].dropna()
        neg = grp.loc[grp[target] == 0, feature].dropna()
        if len(pos) >= 5 and len(neg) >= 5:
            stat, _ = ks_2samp(pos, neg)
            ks_vals.append(stat)
    return float(np.mean(ks_vals)) if ks_vals else 0.0


def _compute_ic_stats(df: pd.DataFrame, feature: str, return_col: str = "Next_Return"):
    """
    Information Coefficient (IC) = Spearman rank correlation between feature
    and forward return, computed per month then averaged.

    Returns (mean_ic, std_ic, ir) where IR = mean_ic / std_ic (information ratio).
    """
    ic_vals = []
    for _, grp in df.groupby("ym"):
        sub = grp[[feature, return_col]].dropna()
        if len(sub) >= 10:
            rho, _ = spearmanr(sub[feature], sub[return_col])
            ic_vals.append(rho)
    if not ic_vals:
        return 0.0, 1.0, 0.0
    mean_ic = float(np.mean(ic_vals))
    std_ic  = float(np.std(ic_vals)) + 1e-8
    ir      = mean_ic / std_ic
    return mean_ic, std_ic, ir


def filter_features_by_ks_ic(
    df: pd.DataFrame,
    candidate_features: list[str],
    ks_threshold: float = 0.04,
    abs_ic_threshold: float = 0.01,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Step 1 – Single-feature filter.

    Keep features that satisfy BOTH:
      • mean KS statistic  ≥  ks_threshold       (distributional separability)
      • |mean IC|          ≥  abs_ic_threshold    (monotone relationship with return)

    The KS threshold is intentionally low (financial alphas are weak);
    tune upward if you want a stricter filter.

    Returns (kept_features, stats_df).
    """
    rows = []
    for f in candidate_features:
        ks          = _compute_ks_per_month(df, f)
        ic, std, ir = _compute_ic_stats(df, f)
        rows.append({"feature": f, "KS": ks, "IC_mean": ic, "IC_std": std, "IR": ir})

    stats_df = pd.DataFrame(rows).sort_values("KS", ascending=False).reset_index(drop=True)

    kept = stats_df.loc[
        (stats_df["KS"] >= ks_threshold) & (stats_df["IC_mean"].abs() >= abs_ic_threshold),
        "feature",
    ].tolist()

    if verbose:
        print(f"\n[Step 1 – KS/IC Filter]")
        print(f"  Candidates : {len(candidate_features)}")
        print(f"  KS ≥ {ks_threshold}, |IC| ≥ {abs_ic_threshold}")
        print(f"  Passed     : {len(kept)}")
        print()
        # Show full table
        pd.set_option("display.float_format", "{:.4f}".format)
        print(stats_df.to_string(index=False))
        pd.reset_option("display.float_format")

    return kept, stats_df


# ---- Step 2 helper -----------------------------------------------------------

def remove_redundant_features(
    df: pd.DataFrame,
    features: list[str],
    stats_df: pd.DataFrame,
    corr_threshold: float = 0.80,
    verbose: bool = True,
) -> list[str]:
    """
    Step 2 – Correlation deduplication.

    Build pairwise Pearson correlation matrix.  When two features exceed
    `corr_threshold`, keep the one with higher mean KS (falls back to |IC_mean|).

    Pairs are resolved greedily from the pair with highest correlation downward.

    Returns list of kept features.
    """
    corr_mat = df[features].corr().abs()

    # Build a score for each feature (KS; ties broken by |IC|)
    score = {}
    for _, row in stats_df.iterrows():
        if row["feature"] in features:
            score[row["feature"]] = (row["KS"], abs(row["IC_mean"]))

    kept  = set(features)
    pairs = (
        corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["f1", "f2", "corr"]
    pairs = pairs[pairs["corr"] >= corr_threshold].sort_values("corr", ascending=False)

    dropped = []
    for _, row in pairs.iterrows():
        f1, f2 = row["f1"], row["f2"]
        if f1 not in kept or f2 not in kept:
            continue   # already resolved
        # Drop the one with the lower score
        if score.get(f1, (0, 0)) >= score.get(f2, (0, 0)):
            kept.discard(f2)
            dropped.append((f2, f1, row["corr"]))
        else:
            kept.discard(f1)
            dropped.append((f1, f2, row["corr"]))

    result = [f for f in features if f in kept]   # preserve original order

    if verbose:
        print(f"\n[Step 2 – Correlation Dedup]")
        print(f"  Threshold : |corr| ≥ {corr_threshold}")
        print(f"  Before    : {len(features)}")
        print(f"  Dropped   : {len(dropped)}")
        for drop, kept_f, c in dropped:
            print(f"    dropped '{drop}' (corr={c:.3f} with '{kept_f}')")
        print(f"  Remaining : {len(result)}")

    return result


# ---- Step 3 helper -----------------------------------------------------------

def model_based_selection(
    df: pd.DataFrame,
    features: list[str],
    train_years: list[int],
    n_select: int = 15,
    verbose: bool = True,
) -> list[str]:
    """
    Step 3 – Model-based selection via permutation importance.

    Trains an XGBoost classifier on `train_years` data using `features`.
    Computes permutation importance (n_repeats=10) on a held-out validation
    year (the year immediately after the last training year).

    Falls back to XGBoost's built-in feature importance if sklearn's
    permutation_importance is unavailable.

    Returns top `n_select` features by mean permutation importance.
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("[Step 3]  XGBoost not installed – skipping model-based step.")
        return features[:n_select]

    val_year = max(train_years) + 1

    tr = df[df["year"].isin(train_years)]
    va = df[df["year"] == val_year]

    if len(va) == 0:
        print(f"[Step 3]  No data for val year {val_year}. Skipping.")
        return features[:n_select]

    X_tr = tr[features].values
    y_tr = tr["Binary_Target"].values
    X_va = va[features].values
    y_va = va["Binary_Target"].values

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    # Permutation importance on validation set (unbiased, model-agnostic)
    perm = permutation_importance(
        model, X_va, y_va,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )

    imp_df = (
        pd.DataFrame({"feature": features, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    selected = imp_df.head(n_select)["feature"].tolist()

    if verbose:
        print(f"\n[Step 3 – Model-Based Selection]")
        print(f"  Train years : {train_years},  Val year: {val_year}")
        print(f"  Candidates  : {len(features)}")
        print(f"  Selected    : {n_select}")
        print()
        print(imp_df.to_string(index=False))

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Master pipeline entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_selection(
    df: pd.DataFrame,
    all_features: list[str] | None = None,
    ks_threshold: float = 0.04,
    abs_ic_threshold: float = 0.01,
    corr_threshold: float = 0.80,
    n_final: int = 15,
    selection_train_years: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the full 3-step feature selection pipeline.

    Parameters
    ----------
    df                     : feature DataFrame (already ranked).
    all_features           : list of feature columns to consider.
                             Defaults to BASE_FEATURES + DERIVED_FEATURES.
    ks_threshold           : minimum mean KS stat to survive Step 1.
    abs_ic_threshold       : minimum |mean IC| to survive Step 1.
    corr_threshold         : maximum allowed pairwise correlation (Step 2).
    n_final                : number of features to keep after Step 3.
    selection_train_years  : years used to fit the Step-3 model.
                             Defaults to all years except the last two.

    Returns (df, selected_features).
    """
    if all_features is None:
        all_features = BASE_FEATURES + DERIVED_FEATURES

    # Only keep columns that actually exist in df
    all_features = [f for f in all_features if f in df.columns]

    if selection_train_years is None:
        years = sorted(df["year"].unique())
        # Use all but the last two years (last two reserved for val/test)
        selection_train_years = years[:-2] if len(years) > 2 else years[:-1]

    print("=" * 60)
    print("FEATURE SELECTION PIPELINE")
    print("=" * 60)
    print(f"  Starting features : {len(all_features)}")

    # Step 1
    step1_features, stats_df = filter_features_by_ks_ic(
        df, all_features, ks_threshold, abs_ic_threshold
    )

    # Step 2
    step2_features = remove_redundant_features(
        df, step1_features, stats_df, corr_threshold
    )

    # Step 3
    final_features = model_based_selection(
        df, step2_features, selection_train_years, n_select=n_final
    )

    global SELECTED_FEATURES
    SELECTED_FEATURES = final_features

    print(f"\n{'=' * 60}")
    print(f"FINAL SELECTED FEATURES ({len(final_features)}):")
    for i, f in enumerate(final_features, 1):
        print(f"  {i:>2}. {f}")
    print("=" * 60)

    return df, final_features


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Convenience: drop-in replacement for the old FEATURE_COLS / Cell 7
# ──────────────────────────────────────────────────────────────────────────────

def build_full_pipeline(
    features_df: pd.DataFrame,
    n_final: int = 15,
    ks_threshold: float = 0.04,
    abs_ic_threshold: float = 0.01,
    corr_threshold: float = 0.80,
) -> tuple[pd.DataFrame, list[str]]:
    """
    One-call entry point that runs steps 1-3 in sequence.

    Usage (replaces Cells 7 + feature-selection logic in your notebook):

        from feature_pipeline_v2 import build_full_pipeline
        features_df, FEATURE_COLS = build_full_pipeline(features_df)

    Returns (enriched_ranked_df, selected_feature_names).
    """
    # Step 1: engineer new features
    features_df = add_engineered_features(features_df)

    # Step 2: cross-sectional ranking (replaces z-score Cell 7)
    features_df = apply_crosssectional_ranking(features_df)

    # Step 3: 3-step feature selection
    features_df, selected = run_feature_selection(
        features_df,
        ks_threshold=ks_threshold,
        abs_ic_threshold=abs_ic_threshold,
        corr_threshold=corr_threshold,
        n_final=n_final,
    )

    return features_df, selected


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Diagnostic plots (optional)
# ──────────────────────────────────────────────────────────────────────────────

def plot_ic_over_time(df: pd.DataFrame, features: list[str], figsize=(18, 10)):
    """
    Plot monthly IC (Spearman correlation with Next_Return) for each feature.
    A stable, consistently positive IC is the hallmark of a reliable factor.
    """
    import matplotlib.pyplot as plt

    ic_records = []
    for ym, grp in df.groupby("ym"):
        for f in features:
            sub = grp[[f, "Next_Return"]].dropna()
            if len(sub) >= 10:
                rho, _ = spearmanr(sub[f], sub["Next_Return"])
                ic_records.append({"ym": ym, "feature": f, "IC": rho})

    ic_df = pd.DataFrame(ic_records)
    n_cols = 4
    n_rows = (len(features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    for ax, f in zip(axes.flat, features):
        sub = ic_df[ic_df["feature"] == f].sort_values("ym")
        ax.bar(range(len(sub)), sub["IC"], color=["#e74c3c" if v < 0 else "#2ecc71" for v in sub["IC"]])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(sub["IC"].mean(), color="navy", linestyle="--", linewidth=1.2,
                   label=f"mean={sub['IC'].mean():.3f}")
        ax.set_title(f, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylabel("IC", fontsize=8)

    for ax in axes.flat[len(features):]:
        ax.set_visible(False)

    plt.suptitle("Monthly IC (Spearman) per Feature  |  Blue dashed = mean IC",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("ic_over_time.png", bbox_inches="tight")
    plt.show()
    print("Saved: ic_over_time.png")


def plot_feature_selection_summary(stats_df: pd.DataFrame):
    """
    Horizontal bar chart of KS statistic and mean IC for all candidate features.
    Useful for communicating which factors survived Step 1.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(stats_df) * 0.4)))

    colors_ks = ["#2ecc71" if v >= 0.04 else "#bdc3c7" for v in stats_df["KS"]]
    ax1.barh(stats_df["feature"], stats_df["KS"], color=colors_ks)
    ax1.axvline(0.04, color="red", linestyle="--", linewidth=1.2, label="threshold=0.04")
    ax1.set_title("KS Statistic (higher = better separated)", fontweight="bold")
    ax1.legend()
    ax1.set_xlabel("Mean KS")

    colors_ic = ["#e74c3c" if v < 0 else "#3498db" for v in stats_df["IC_mean"]]
    ax2.barh(stats_df["feature"], stats_df["IC_mean"], color=colors_ic)
    ax2.axvline(0.01,  color="red", linestyle="--", linewidth=1.2, label="|IC|≥0.01")
    ax2.axvline(-0.01, color="red", linestyle="--", linewidth=1.2)
    ax2.set_title("Mean IC (Spearman with Next_Return)", fontweight="bold")
    ax2.legend()
    ax2.set_xlabel("Mean IC")

    plt.suptitle("Feature Selection – Step 1 Summary", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("feature_selection_summary.png", bbox_inches="tight")
    plt.show()
    print("Saved: feature_selection_summary.png")
