"""
03_Exploratory_Data_Analysis/spectral_analysis.py

Dedicated Standalone Executable for Deep Hyperspectral Characterization:
1. Loads cleaned reflectance dataset from 02_Data_Preprocessing.
2. Extracts and profiles continuous mean signatures (Control, Infected, Genotypes, Watering).
3. Evaluates biophysical spectral domains (VIS, Red-Edge, NIR, SWIR-1, SWIR-2).
4. Computes difference spectra (Delta R, Delta R_rel) and statistical significance.
5. Generates publication-quality spectral visualizations.
"""

import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.append(str(MODULE_DIR))

from utils.eda_utils import setup_logger, load_config, load_dataset, save_json
from analysis.spectral_analysis import HyperspectralSignatureAnalyzer
from analysis.disease_analysis import DiseaseSpectralComparator
from visualizations.spectral_plots import (
    plot_global_spectral_signature,
    plot_spectral_regions_breakdown,
    plot_spectral_variability,
)
from visualizations.disease_comparison_plots import (
    plot_healthy_vs_diseased_signatures,
    plot_difference_spectra,
    plot_wavelength_significance_manhattan,
    plot_cohens_d_effect_size_spectrum,
)

logger = setup_logger("spectral_analysis_cli")


def run_spectral_analysis_pipeline(config_path: Path = None) -> bool:
    """Executes dedicated hyperspectral analysis pipeline."""
    config = load_config(config_path)
    logger.info("=" * 70)
    logger.info("TERRASPECTRA: DEDICATED HYPERSPECTRAL ANALYSIS PIPELINE")
    logger.info("=" * 70)

    # 1. Locate dataset
    data_path = PROJECT_ROOT / config["data_inputs"]["cleaned_data_path"]
    if not data_path.exists():
        data_path = PROJECT_ROOT / config["data_inputs"]["cleaned_data_csv"]
    if not data_path.exists():
        logger.error(f"Cleaned dataset not found at: {data_path}")
        return False

    logger.info(f"Loading hyperspectral dataset from: {data_path.name}...")
    metadata_df, spectral_matrix, wavelengths, spectral_cols = load_dataset(data_path)
    logger.info(f"Loaded {spectral_matrix.shape[0]} samples across {len(wavelengths)} bands ({wavelengths.min():.2f} - {wavelengths.max():.2f} nm).")

    out_base = MODULE_DIR / "outputs"
    spec_dir = out_base / "spectral_analysis"
    disease_dir = out_base / "disease_analysis"
    viz_dir = out_base / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 2. Spectral Signatures & Regions
    logger.info("Analyzing continuous spectral signatures and biophysical domains...")
    sig_analyzer = HyperspectralSignatureAnalyzer(metadata_df, spectral_matrix, wavelengths)
    sig_summary = sig_analyzer.run_spectral_analysis(config, spec_dir)

    # 3. Disease Comparative Analysis
    logger.info("Evaluating healthy vs. diseased difference spectra and significance...")
    disease_comparator = DiseaseSpectralComparator(metadata_df, spectral_matrix, wavelengths)
    disease_summary = disease_comparator.run_disease_analysis(config, disease_dir)

    # 4. Generate Visualizations
    logger.info("Generating publication-grade spectral visualizations...")
    plot_global_spectral_signature(spectral_matrix, wavelengths, viz_dir / "03_global_spectral_signature.png")
    plot_spectral_regions_breakdown(spectral_matrix, wavelengths, config, viz_dir / "04_spectral_regions_breakdown.png")
    plot_spectral_variability(spectral_matrix, wavelengths, viz_dir / "spectral_variability_cv.png")

    diff_df = disease_comparator.compute_difference_spectra(config["data_inputs"].get("target_disease_column", "disease"))
    sig_df = disease_comparator.run_significance_tests(config["data_inputs"].get("target_disease_column", "disease"))
    plot_healthy_vs_diseased_signatures(diff_df, viz_dir / "05_healthy_vs_diseased_signatures.png")
    plot_difference_spectra(diff_df, viz_dir / "06_difference_spectra.png")
    plot_wavelength_significance_manhattan(sig_df, viz_dir / "07_wavelength_significance_manhattan.png")
    plot_cohens_d_effect_size_spectrum(sig_df, viz_dir / "08_cohens_d_effect_size_spectrum.png")

    logger.info("=" * 70)
    logger.info(" ✅ DEDICATED HYPERSPECTRAL ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info(f"  * Bands Analyzed    : {len(wavelengths)} continuous narrow bands")
    logger.info(f"  * FDR Significant   : {disease_summary['fdr_significant_bands_count']} bands (at q < 0.05)")
    logger.info(f"  * Outputs Directory : {spec_dir}")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    success = run_spectral_analysis_pipeline()
    sys.exit(0 if success else 1)
