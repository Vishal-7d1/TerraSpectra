"""
04_Feature_Engineering/feature_selection.py

Standalone Leakage-Free Feature Selection & Dimensionality Reduction CLI:
1. Performs Stratified Group Splitting (grouped by barcode, stratified by disease).
2. Fits variance filtering, correlation filtering, and Random Forest feature selection strictly on Train.
3. Fits PCA strictly on Train and transforms Train, Val, and Test.
4. Serializes fitted PCA model, feature datasets, split assignments, and importance reports.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.append(str(MODULE_DIR))

from utils.feature_utils import (
    setup_logger,
    load_config,
    save_dataframe,
    save_json,
)
from feature_extraction import run_feature_extraction
from validation.split_generator import StratifiedGroupSplitter
from validation.leakage_check import DataLeakageAuditor
from validation.feature_validation import FeatureQualityAuditor
from selection.variance_filter import LeakageFreeVarianceFilter
from selection.correlation_filter import LeakageFreeCorrelationFilter
from selection.model_importance import LeakageFreeModelImportanceSelector
from selection.mutual_information import LeakageFreeMutualInfoSelector
from selection.statistical_selection import LeakageFreeStatisticalSelector
from dimensionality_reduction.pca import LeakageFreePCA

logger = setup_logger("feature_selection_cli")


def run_feature_selection(config_path: Path = None) -> bool:
    """Executes the complete leakage-safe feature selection and PCA pipeline."""
    config = load_config(config_path)
    logger.info("=" * 70)
    logger.info("TERRASPECTRA: LEAKAGE-FREE FEATURE SELECTION & REDUCTION PIPELINE")
    logger.info("=" * 70)

    # 1. Check or run feature extraction
    extracted_pq = MODULE_DIR / "outputs" / "extracted_features" / "extracted_features.parquet"
    if not extracted_pq.exists():
        logger.info("Extracted features not found. Running feature extraction stage...")
        metadata_df, engineered_df, _ = run_feature_extraction(config_path)
    else:
        logger.info(f"Loading pre-extracted features from {extracted_pq.name}...")
        full_df = pd.read_parquet(extracted_pq)
        meta_cols = [
            "Unnamed: 0", "barcode", "Plant Species", "Replicate", "Lane", "Position",
            "disease", "watering", "genotype ID", "value", "swir", "vnir", "date-time",
            "dead", "disease_label_binary"
        ]
        present_meta = [c for c in meta_cols if c in full_df.columns]
        metadata_df = full_df[present_meta].copy()
        feat_cols = [c for c in full_df.columns if c not in present_meta]
        engineered_df = full_df[feat_cols].copy()

    # 2. Quality Audit on Raw Extracted Features
    logger.info("Auditing quality of extracted features...")
    quality_auditor = FeatureQualityAuditor(engineered_df)
    quality_report = quality_auditor.audit_quality()
    reports_dir = MODULE_DIR / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_json(quality_report, reports_dir / "feature_quality_report.json")

    # 3. Stratified Group Partitioning (Strictly by plant barcode)
    logger.info("Generating biologically isolated Stratified Group Splits (70% Train, 15% Val, 15% Test)...")
    group_col = config["data_inputs"].get("group_column", "barcode")
    target_col = config["data_inputs"].get("target_column", "disease")

    splitter = StratifiedGroupSplitter(
        group_col=group_col,
        target_col=target_col,
        random_state=config["splitting"].get("random_state", 42),
    )
    split_df = splitter.generate_splits(
        metadata_df,
        train_ratio=config["splitting"].get("train_ratio", 0.70),
        val_ratio=config["splitting"].get("val_ratio", 0.15),
        test_ratio=config["splitting"].get("test_ratio", 0.15),
    )
    split_audit = splitter.audit_splits(split_df)

    # 4. Rigorous Leakage Audit
    logger.info("Executing defensive data leakage verification...")
    leakage_audit = DataLeakageAuditor.audit_group_isolation(split_df, group_col=group_col)
    if not leakage_audit["is_completely_isolated"]:
        raise ValueError(f"FATAL: Biological specimen leakage detected across splits: {leakage_audit}")
    logger.info("Data Leakage Audit: PASSED (Zero specimen overlap between Train, Val, and Test).")

    splits_dir = MODULE_DIR / "outputs" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(splits_dir / "split_assignments.csv", index=False)
    save_json({"split_audit": split_audit, "leakage_audit": leakage_audit}, splits_dir / "split_metadata.json")

    # 5. Extract Train, Val, Test Subsets
    train_mask = (split_df["split_partition"] == "train").values
    val_mask = (split_df["split_partition"] == "val").values
    test_mask = (split_df["split_partition"] == "test").values

    X_train_raw = engineered_df.loc[train_mask].copy()
    y_train = (metadata_df.loc[train_mask, target_col] == config["data_inputs"].get("positive_class", "Infected")).astype(int).values

    # 6. Leakage-Free Feature Selection Pipeline (Fit strictly on Train!)
    logger.info(f"Initiating feature selection cascade on training split (N={len(X_train_raw)} observations)...")
    initial_feat_count = X_train_raw.shape[1]

    # Stage 1: Variance Filter
    var_filter = LeakageFreeVarianceFilter(threshold=config["selection"].get("variance_threshold", 1e-5))
    X_train_v = var_filter.fit_transform(X_train_raw)
    after_var_count = X_train_v.shape[1]
    logger.info(f"Stage 1 (Variance Filter): Retained {after_var_count}/{initial_feat_count} features.")

    # Stage 2: Correlation Filter
    corr_filter = LeakageFreeCorrelationFilter(threshold=config["selection"].get("correlation_threshold", 0.95))
    X_train_c = corr_filter.fit_transform(X_train_v, y_train)
    after_corr_count = X_train_c.shape[1]
    logger.info(f"Stage 2 (Correlation Filter): Retained {after_corr_count}/{after_var_count} non-collinear features.")

    # Stage 3: Random Forest Feature Importance Selection
    target_k = min(config["selection"].get("target_selected_features", 35), after_corr_count)
    rf_selector = LeakageFreeModelImportanceSelector(
        n_estimators=config["selection"].get("rf_n_estimators", 100),
        top_k=target_k,
        random_state=config["selection"].get("random_state", 42),
    )
    X_train_selected = rf_selector.fit_transform(X_train_c, y_train)
    final_selected_count = X_train_selected.shape[1]
    logger.info(f"Stage 3 (Model Importance Selection): Finalized {final_selected_count} top predictive features.")

    # Apply fitted selectors to full dataset (and val/test)
    X_full_selected = rf_selector.transform(corr_filter.transform(var_filter.transform(engineered_df)))

    # Save Selected Features
    selected_dir = MODULE_DIR / "outputs" / "selected_features"
    selected_dir.mkdir(parents=True, exist_ok=True)
    full_selected_with_meta = pd.concat([
        metadata_df.reset_index(drop=True),
        split_df[["split_partition"]].reset_index(drop=True),
        X_full_selected.reset_index(drop=True)
    ], axis=1)
    save_dataframe(full_selected_with_meta, selected_dir / "selected_features", save_csv=True, save_parquet=True)
    save_dataframe(X_full_selected, selected_dir / "selected_features_only", save_csv=True, save_parquet=True)

    # 7. Model Importance Rankings Export
    importance_dir = MODULE_DIR / "outputs" / "feature_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)
    importance_df = rf_selector.ranking_df_
    importance_df.to_csv(importance_dir / "feature_importance.csv", index=False)

    # Also compute MI and F-statistic for triangulation
    mi_selector = LeakageFreeMutualInfoSelector(top_k=target_k).fit(X_train_c, y_train)
    f_selector = LeakageFreeStatisticalSelector(top_k=target_k).fit(X_train_c, y_train)
    mi_selector.ranking_df_.to_csv(importance_dir / "mutual_information_ranking.csv", index=False)
    f_selector.ranking_df_.to_csv(importance_dir / "statistical_f_score_ranking.csv", index=False)

    importance_summary = {
        "top_features_by_random_forest": list(importance_df["feature"].head(10).values),
        "top_features_by_mutual_info": list(mi_selector.ranking_df_["feature"].head(10).values),
        "top_features_by_f_score": list(f_selector.ranking_df_["feature"].head(10).values),
    }
    save_json(importance_summary, importance_dir / "importance_summary.json")

    # 8. Unsupervised Dimensionality Reduction (PCA on Train Split)
    logger.info("Fitting PCA dimensionality reduction strictly on training partition...")
    pca_dir = MODULE_DIR / "outputs" / "dimensionality_reduction"
    pca_dir.mkdir(parents=True, exist_ok=True)

    pca_transformer = LeakageFreePCA(
        n_components=config["dimensionality_reduction"].get("pca_n_components", 10),
        random_state=config["dimensionality_reduction"].get("random_state", 42),
    )
    pca_transformer.fit(X_train_raw)
    pca_scores_full = pca_transformer.transform(engineered_df)
    pca_transformer.save_fitted_pipeline(pca_dir / "fitted_pca.joblib")

    full_reduced_with_meta = pd.concat([
        metadata_df.reset_index(drop=True),
        split_df[["split_partition"]].reset_index(drop=True),
        pca_scores_full.reset_index(drop=True)
    ], axis=1)
    save_dataframe(full_reduced_with_meta, pca_dir / "reduced_features", save_csv=True, save_parquet=True)
    save_dataframe(pca_scores_full, pca_dir / "reduced_features_only", save_csv=True, save_parquet=True)

    # 9. Selection Summary Report
    selection_report = {
        "initial_feature_count": initial_feat_count,
        "after_variance_filter_count": after_var_count,
        "variance_dropped_count": len(var_filter.dropped_columns),
        "after_correlation_filter_count": after_corr_count,
        "correlation_dropped_count": len(corr_filter.dropped_columns),
        "final_selected_features_count": final_selected_count,
        "selected_feature_names": list(X_full_selected.columns),
        "pca_components_fitted": pca_transformer.n_components,
        "pca_cumulative_variance_top2": round(float(np.sum(pca_transformer.pca.explained_variance_ratio_[:2]) * 100), 2),
        "pca_cumulative_variance_total": round(float(np.sum(pca_transformer.pca.explained_variance_ratio_) * 100), 2),
    }
    save_json(selection_report, reports_dir / "feature_selection_report.json")

    logger.info("=" * 70)
    logger.info(" ✅ FEATURE SELECTION & REDUCTION COMPLETED SUCCESSFULLY")
    logger.info(f"  * Selected Features   : {final_selected_count} features")
    logger.info(f"  * PCA Components      : {pca_transformer.n_components} components")
    logger.info(f"  * Zero-Leakage Splits : Train ({split_audit['train']['total_observations']}), Val ({split_audit['val']['total_observations']}), Test ({split_audit['test']['total_observations']})")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    run_feature_selection()
