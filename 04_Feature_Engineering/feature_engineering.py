"""
04_Feature_Engineering/feature_engineering.py

MASTER PIPELINE ORCHESTRATOR FOR TERRASPECTRA:
Hyperspectral Crop Disease Forecasting - Feature Engineering Module

This orchestrator executes the entire end-to-end feature engineering workflow:
1. Validates inputs from 02_Data_Preprocessing (physical reflectance).
2. Computes 51 domain-specific engineered features:
   - 10 narrow-band physiological & vegetation indices (NDVI, NDRE, PRI, etc.)
   - 8 spectral derivative features (dR/dlambda, REP, REP amplitude, REP area, slopes)
   - 11 spectral shape & absorption features (AUC, continuum water depths at 1450 & 1940 nm)
   - 22 global and domain-stratified statistical moments across VIS, Red-Edge, NIR, SWIR1, SWIR2.
3. Performs rigorous Quality Audit on extracted features (NaN, Inf, constant checks).
4. Performs Stratified Group Splitting (70% Train, 15% Val, 15% Test) strictly grouped by plant barcode.
5. Performs Defensive Data Leakage Verification (0% barcode overlap).
6. Executes Leakage-Free Feature Selection Cascade (Fit ONLY on Train):
   - Variance Filter
   - Correlation Filter (|r| < threshold)
   - Random Forest Gini Importance Selection
   - Mutual Information and ANOVA F-statistic triangulation
7. Fits PCA Dimensionality Reduction strictly on Train, transforms Train/Val/Test, and serializes pipeline.
8. Generates publication-grade scientific visualizations.
9. Computes SHA-256 checksums and produces an authoritative execution manifest.
"""

import sys
import time
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
    load_dataset,
    save_dataframe,
    save_json,
    calculate_sha256,
)
from extraction.vegetation_indices import VegetationIndexExtractor
from extraction.derivative_features import SpectralDerivativeExtractor
from extraction.spectral_features import SpectralShapeExtractor
from extraction.statistical_features import StatisticalFeatureExtractor
from validation.split_generator import StratifiedGroupSplitter
from validation.leakage_check import DataLeakageAuditor
from validation.feature_validation import FeatureQualityAuditor
from selection.variance_filter import LeakageFreeVarianceFilter
from selection.correlation_filter import LeakageFreeCorrelationFilter
from selection.model_importance import LeakageFreeModelImportanceSelector
from selection.mutual_information import LeakageFreeMutualInfoSelector
from selection.statistical_selection import LeakageFreeStatisticalSelector
from dimensionality_reduction.pca import LeakageFreePCA
from visualizations.feature_plots import (
    plot_feature_importance_ranking,
    plot_extracted_feature_correlation_heatmap,
    plot_spectral_derivatives_profile,
    plot_pca_scree_and_variance,
    plot_feature_selection_stages,
)

logger = setup_logger("feature_engineering_pipeline")


def execute_pipeline(config_path: Path = None) -> Dict[str, Any]:
    """Runs the complete end-to-end TerraSpectra feature engineering pipeline."""
    start_time = time.time()
    config = load_config(config_path)

    logger.info("=" * 80)
    logger.info("🚀 TERRASPECTRA: COMPLETE FEATURE ENGINEERING MASTER PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Root Directory: {MODULE_DIR}")

    # Output directories
    out_dirs = {
        "extracted": MODULE_DIR / "outputs" / "extracted_features",
        "selected": MODULE_DIR / "outputs" / "selected_features",
        "pca": MODULE_DIR / "outputs" / "dimensionality_reduction",
        "importance": MODULE_DIR / "outputs" / "feature_importance",
        "splits": MODULE_DIR / "outputs" / "splits",
        "reports": MODULE_DIR / "outputs" / "reports",
        "vis": MODULE_DIR / "outputs" / "visualizations",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STAGE 1: LOAD PREPROCESSED DATASET
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 1/8] LOADING PREPROCESSED HYPERSPECTRAL DATA ---")
    data_path = PROJECT_ROOT / config["data_inputs"]["cleaned_data_path"]
    if not data_path.exists():
        data_path = PROJECT_ROOT / config["data_inputs"]["cleaned_data_csv"]
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned preprocessed dataset not found at: {data_path}")

    metadata_df, spectral_matrix, wavelengths, meta_cols = load_dataset(data_path)
    logger.info(f"Loaded {len(metadata_df)} observations across {len(wavelengths)} contiguous spectral bands ({wavelengths[0]:.2f} nm to {wavelengths[-1]:.2f} nm).")
    logger.info(f"Metadata columns identified: {len(meta_cols)} columns.")

    # ---------------------------------------------------------
    # STAGE 2: DOMAIN FEATURE EXTRACTION
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 2/8] SCIENTIFIC FEATURE EXTRACTION ---")
    # 2a. Vegetation Indices
    logger.info("Computing 10 narrow-band vegetation indices (NDVI, NDRE, PRI, mSR705, WBI, CRI, PSRI, GNDVI, MCARI, SIPI)...")
    vi_ext = VegetationIndexExtractor(spectral_matrix, wavelengths)
    vi_df, vi_meta = vi_ext.extract_indices()

    # 2b. Derivative Features
    logger.info("Computing 1st & 2nd spectral derivatives, Red-Edge position (REP), and water absorption slopes...")
    deriv_ext = SpectralDerivativeExtractor(spectral_matrix, wavelengths)
    deriv_df, deriv_meta = deriv_ext.extract_derivative_features()

    # 2c. Spectral Shape Features
    logger.info("Computing continuum-relative absorption depths (1450 & 1940 nm), AUC, and spectral slopes...")
    shape_ext = SpectralShapeExtractor(spectral_matrix, wavelengths)
    shape_df, shape_meta = shape_ext.extract_shape_features()

    # 2d. Statistical Features
    logger.info("Computing global and region-stratified statistical moments (VIS, Red-Edge, NIR, SWIR1, SWIR2)...")
    stat_ext = StatisticalFeatureExtractor(spectral_matrix, wavelengths)
    stat_df, stat_meta = stat_ext.extract_statistical_features()

    engineered_df = pd.concat([vi_df, deriv_df, shape_df, stat_df], axis=1)
    logger.info(f"Feature Extraction Completed: {engineered_df.shape[1]} engineered features extracted across {engineered_df.shape[0]} observations.")

    # Save extracted features
    full_extracted_df = pd.concat([metadata_df.reset_index(drop=True), engineered_df.reset_index(drop=True)], axis=1)
    save_dataframe(full_extracted_df, out_dirs["extracted"] / "extracted_features", save_csv=True, save_parquet=True)
    save_dataframe(engineered_df, out_dirs["extracted"] / "engineered_features_only", save_csv=True, save_parquet=True)

    extracted_meta = {
        "total_samples": int(len(engineered_df)),
        "total_extracted_features": int(engineered_df.shape[1]),
        "category_counts": {
            "vegetation_indices": len(vi_df.columns),
            "derivative_features": len(deriv_df.columns),
            "spectral_shape_features": len(shape_df.columns),
            "statistical_features": len(stat_df.columns),
        },
        "feature_names_by_category": {
            "vegetation_indices": list(vi_df.columns),
            "derivative_features": list(deriv_df.columns),
            "spectral_shape_features": list(shape_df.columns),
            "statistical_features": list(stat_df.columns),
        }
    }
    save_json(extracted_meta, out_dirs["extracted"] / "extracted_features_metadata.json")

    # ---------------------------------------------------------
    # STAGE 3: DATA QUALITY AUDIT
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 3/8] FEATURE MATRIX QUALITY AUDIT ---")
    quality_auditor = FeatureQualityAuditor(engineered_df)
    quality_report = quality_auditor.audit_quality()
    save_json(quality_report, out_dirs["reports"] / "feature_quality_report.json")
    if not quality_report.get("is_valid", False):
        logger.warning(f"Quality Auditor flagged issues: NaNs={quality_report.get('total_missing_values')}, Infs={quality_report.get('total_infinite_values')}, Constants={quality_report.get('constant_features_count')}")
    else:
        logger.info("Feature Quality Audit: 100% CLEAN (0 NaNs, 0 Infs, 0 constant features).")

    # ---------------------------------------------------------
    # STAGE 4: STRATIFIED GROUP PARTITIONING (LEAKAGE DEFENSE)
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 4/8] BIOLOGICAL GROUPED SPLITTING (ZERO LEAKAGE) ---")
    group_col = config["data_inputs"].get("group_column", "barcode")
    target_col = config["data_inputs"].get("target_column", "disease")
    train_ratio = config["splitting"].get("train_ratio", 0.70)
    val_ratio = config["splitting"].get("val_ratio", 0.15)
    test_ratio = config["splitting"].get("test_ratio", 0.15)

    splitter = StratifiedGroupSplitter(
        group_col=group_col,
        target_col=target_col,
        random_state=config["splitting"].get("random_state", 42),
    )
    split_df = splitter.generate_splits(metadata_df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    split_audit = splitter.audit_splits(split_df)

    # Rigorous leakage audit
    leakage_audit = DataLeakageAuditor.audit_group_isolation(split_df, group_col=group_col)
    if not leakage_audit["is_completely_isolated"]:
        raise ValueError(f"FATAL DATA LEAKAGE: Biological barcodes overlap between partitions: {leakage_audit}")

    logger.info("Data Leakage Audit: PASSED! Zero specimen overlap between Train, Val, and Test partitions.")
    logger.info(f"Split breakdown: Train={split_audit['train']['total_observations']} scans ({split_audit['train']['unique_plant_barcodes']} plants), "
                f"Val={split_audit['val']['total_observations']} scans ({split_audit['val']['unique_plant_barcodes']} plants), "
                f"Test={split_audit['test']['total_observations']} scans ({split_audit['test']['unique_plant_barcodes']} plants).")

    split_df.to_csv(out_dirs["splits"] / "split_assignments.csv", index=False)
    save_json({"split_audit": split_audit, "leakage_audit": leakage_audit}, out_dirs["splits"] / "split_metadata.json")

    # ---------------------------------------------------------
    # STAGE 5: LEAKAGE-FREE FEATURE SELECTION CASCADE
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 5/8] MULTI-STAGE FEATURE SELECTION CASCADE ---")
    train_mask = (split_df["split_partition"] == "train").values
    val_mask = (split_df["split_partition"] == "val").values
    test_mask = (split_df["split_partition"] == "test").values

    X_train_raw = engineered_df.loc[train_mask].copy()
    y_train = (metadata_df.loc[train_mask, target_col] == config["data_inputs"].get("positive_class", "Infected")).astype(int).values

    initial_features_count = X_train_raw.shape[1]

    # Stage 5.1: Variance Filter
    var_filter = LeakageFreeVarianceFilter(threshold=config["selection"].get("variance_threshold", 1e-5))
    X_train_var = var_filter.fit_transform(X_train_raw)
    after_var_count = X_train_var.shape[1]
    logger.info(f"Step 5.1 (Variance Filter)      : {initial_features_count} -> {after_var_count} features (dropped {len(var_filter.dropped_columns)})")

    # Stage 5.2: Correlation Filter
    corr_filter = LeakageFreeCorrelationFilter(threshold=config["selection"].get("correlation_threshold", 0.95))
    X_train_corr = corr_filter.fit_transform(X_train_var, y_train)
    after_corr_count = X_train_corr.shape[1]
    logger.info(f"Step 5.2 (Correlation Filter)   : {after_var_count} -> {after_corr_count} features (dropped {len(corr_filter.dropped_columns)})")

    # Stage 5.3: Random Forest Feature Importance Selector
    target_k = min(config["selection"].get("target_selected_features", 35), after_corr_count)
    rf_selector = LeakageFreeModelImportanceSelector(
        n_estimators=config["selection"].get("rf_n_estimators", 100),
        top_k=target_k,
        random_state=config["selection"].get("random_state", 42),
    )
    X_train_selected = rf_selector.fit_transform(X_train_corr, y_train)
    final_selected_count = X_train_selected.shape[1]
    logger.info(f"Step 5.3 (Random Forest Select) : {after_corr_count} -> {final_selected_count} top features")

    # Apply fitted pipeline to full feature dataset
    X_full_selected = rf_selector.transform(corr_filter.transform(var_filter.transform(engineered_df)))

    # Save selected features
    full_selected_with_meta = pd.concat([
        metadata_df.reset_index(drop=True),
        split_df[["split_partition"]].reset_index(drop=True),
        X_full_selected.reset_index(drop=True)
    ], axis=1)
    save_dataframe(full_selected_with_meta, out_dirs["selected"] / "selected_features", save_csv=True, save_parquet=True)
    save_dataframe(X_full_selected, out_dirs["selected"] / "selected_features_only", save_csv=True, save_parquet=True)

    # Stage 5.4: Multi-Method Triangulation (MI & ANOVA F-score)
    mi_selector = LeakageFreeMutualInfoSelector(top_k=target_k).fit(X_train_corr, y_train)
    f_selector = LeakageFreeStatisticalSelector(top_k=target_k).fit(X_train_corr, y_train)

    rf_selector.ranking_df_.to_csv(out_dirs["importance"] / "feature_importance.csv", index=False)
    mi_selector.ranking_df_.to_csv(out_dirs["importance"] / "mutual_information_ranking.csv", index=False)
    f_selector.ranking_df_.to_csv(out_dirs["importance"] / "statistical_f_score_ranking.csv", index=False)

    importance_summary = {
        "top_features_random_forest": list(rf_selector.ranking_df_["feature"].head(10).values),
        "top_features_mutual_info": list(mi_selector.ranking_df_["feature"].head(10).values),
        "top_features_anova_f": list(f_selector.ranking_df_["feature"].head(10).values),
    }
    save_json(importance_summary, out_dirs["importance"] / "importance_summary.json")

    selection_report = {
        "initial_feature_count": initial_features_count,
        "after_variance_filter_count": after_var_count,
        "variance_dropped_count": len(var_filter.dropped_columns),
        "after_correlation_filter_count": after_corr_count,
        "correlation_dropped_count": len(corr_filter.dropped_columns),
        "final_selected_features_count": final_selected_count,
        "selected_feature_names": list(X_full_selected.columns),
    }
    save_json(selection_report, out_dirs["reports"] / "feature_selection_report.json")

    # ---------------------------------------------------------
    # STAGE 6: PCA DIMENSIONALITY REDUCTION (FIT ON TRAIN)
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 6/8] PCA DIMENSIONALITY REDUCTION (LEAKAGE FREE) ---")
    pca_k = config["dimensionality_reduction"].get("pca_n_components", 10)
    pca_pipeline = LeakageFreePCA(
        n_components=pca_k,
        random_state=config["dimensionality_reduction"].get("random_state", 42),
    )
    pca_pipeline.fit(X_train_raw)
    pca_scores_full = pca_pipeline.transform(engineered_df)
    pca_pipeline.save_fitted_pipeline(out_dirs["pca"] / "fitted_pca.joblib")

    full_reduced_with_meta = pd.concat([
        metadata_df.reset_index(drop=True),
        split_df[["split_partition"]].reset_index(drop=True),
        pca_scores_full.reset_index(drop=True)
    ], axis=1)
    save_dataframe(full_reduced_with_meta, out_dirs["pca"] / "reduced_features", save_csv=True, save_parquet=True)
    save_dataframe(pca_scores_full, out_dirs["pca"] / "reduced_features_only", save_csv=True, save_parquet=True)

    explained_var = pca_pipeline.pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(explained_var)
    logger.info(f"Fitted PCA ({pca_k} components): Top 2 components explain {cum_var[1]:.2f}%, total explains {cum_var[-1]:.2f}% of variance.")

    # ---------------------------------------------------------
    # STAGE 7: PUBLICATION-GRADE VISUALIZATIONS
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 7/8] GENERATING PUBLICATION VISUALIZATIONS ---")
    dpi = config.get("outputs", {}).get("plot_dpi", 300)

    # 7a. Feature Importance Horizontal Bar Plot
    plot_feature_importance_ranking(
        rf_selector.ranking_df_,
        out_dirs["vis"] / "feature_importance_ranking.png",
        top_k=25,
        dpi=dpi,
    )

    # 7b. Correlation Heatmaps
    plot_extracted_feature_correlation_heatmap(
        engineered_df[list(engineered_df.columns[:25])],
        out_dirs["vis"] / "engineered_features_correlation_heatmap.png",
        dpi=dpi,
    )
    plot_extracted_feature_correlation_heatmap(
        X_full_selected,
        out_dirs["vis"] / "selected_features_correlation_heatmap.png",
        dpi=dpi,
    )

    # 7c. Spectral 1st Derivative Profiles
    plot_spectral_derivatives_profile(
        deriv_ext.d1_matrix,
        wavelengths,
        metadata_df["disease"].values,
        out_dirs["vis"] / "spectral_derivatives_profile.png",
        dpi=dpi,
    )

    # 7d. PCA Scree & Cumulative Variance
    plot_pca_scree_and_variance(
        pca_pipeline.pca,
        out_dirs["vis"] / "pca_scree_and_cumulative_variance.png",
        dpi=dpi,
    )

    # 7e. Feature Selection Funnel
    funnel_counts = {
        "Initial Extracted": initial_features_count,
        "Variance Filter": after_var_count,
        "Correlation Filter": after_corr_count,
        "Random Forest Top-K": final_selected_count,
    }
    plot_feature_selection_stages(
        funnel_counts,
        out_dirs["vis"] / "feature_selection_stages_funnel.png",
        dpi=dpi,
    )
    logger.info("Visualizations successfully generated in outputs/visualizations/.")

    # ---------------------------------------------------------
    # STAGE 8: AUDIT MANIFEST & SHA-256 CHECKSUMS
    # ---------------------------------------------------------
    logger.info("\n--- [STAGE 8/8] COMPILING AUDIT MANIFEST & SHA-256 HASHES ---")
    manifest_files = {}
    for output_subdir in out_dirs.values():
        for f in output_subdir.rglob("*"):
            if f.is_file():
                rel_path = str(f.relative_to(MODULE_DIR)).replace("\\", "/")
                manifest_files[rel_path] = {
                    "size_bytes": f.stat().st_size,
                    "sha256": calculate_sha256(f),
                }

    pipeline_summary = {
        "status": "SUCCESS",
        "pipeline_version": "1.0.0",
        "execution_duration_seconds": round(time.time() - start_time, 2),
        "total_samples": len(engineered_df),
        "total_extracted_features": initial_features_count,
        "selected_features_count": final_selected_count,
        "pca_components_count": pca_k,
        "split_counts": {
            "train": int(split_audit["train"]["total_observations"]),
            "val": int(split_audit["val"]["total_observations"]),
            "test": int(split_audit["test"]["total_observations"]),
        },
        "biological_specimen_isolation_verified": leakage_audit["is_completely_isolated"],
        "top_5_predictive_features": list(rf_selector.ranking_df_["feature"].head(5).values),
        "generated_artifacts_count": len(manifest_files),
        "artifact_manifest": manifest_files,
    }

    save_json(pipeline_summary, out_dirs["reports"] / "feature_engineering_pipeline_report.json")
    save_json(manifest_files, out_dirs["reports"] / "artifact_manifest_sha256.json")

    logger.info("=" * 80)
    logger.info(" ✨ FEATURE ENGINEERING MASTER PIPELINE COMPLETED SUCCESSFULLY! ✨")
    logger.info(f"  * Runtime                    : {pipeline_summary['execution_duration_seconds']} seconds")
    logger.info(f"  * Extracted Features         : {initial_features_count}")
    logger.info(f"  * Final Selected Features    : {final_selected_count}")
    logger.info(f"  * Top-1 Feature (RF Gini)    : {rf_selector.ranking_df_['feature'].iloc[0]}")
    logger.info(f"  * Zero-Leakage Group Splits  : Train ({split_audit['train']['total_observations']}), Val ({split_audit['val']['total_observations']}), Test ({split_audit['test']['total_observations']})")
    logger.info(f"  * Artifacts Generated        : {len(manifest_files)} files with verified SHA-256 hashes")
    logger.info("=" * 80)

    return pipeline_summary


if __name__ == "__main__":
    execute_pipeline()
