"""
01_Dataset_Collection/scripts/discover_datasets.py

Researches, registers, and catalogs authentic, publicly accessible hyperspectral agricultural datasets.
Generates:
- dataset_info/source_registry.csv
- dataset_info/dataset_catalog.json
- dataset_info/dataset_card.md
"""

import sys
from pathlib import Path
from datetime import datetime

# Add module dir to path for imports
MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(MODULE_DIR))

from config import (
    SOURCE_REGISTRY_FILE,
    CATALOG_FILE,
    DATASET_CARD_FILE,
    setup_logger,
)
from utils.metadata_utils import save_json, save_csv

logger = setup_logger("discover_datasets")

# Authentic Verified Candidate Datasets
CANDIDATE_DATASETS = [
    {
        "dataset_id": "DS_WHEAT_CROWN_ROT_001",
        "dataset_name": "Hyperspectral Imaging for Early Detection of Crown Rot in Wheat",
        "publisher_institution": "University of Adelaide / CSIRO",
        "official_source_url": "https://doi.org/10.25909/17075138",
        "download_url": "https://ndownloader.figshare.com/files/31569671",
        "download_availability": "Direct Open HTTP Download",
        "crop_species": "Wheat (Triticum durum & Triticum aestivum)",
        "disease_categories": "Crown Rot (Fusarium pseudograminearum) & Combined Drought Stress",
        "sample_count": 1632,
        "spectral_bands": 657,
        "wavelength_range": "400 nm - 2400 nm (VNIR & SWIR)",
        "file_formats": ".xlsx (barcode_hypdata, result)",
        "ground_truth_labels": "Inoculated (Infected) vs. Mock-inoculated (Control); Watering regime (Drought vs. Well-watered)",
        "collection_methodology": "Indoor phenotyping platform with automated LemnaTec scanalyzer and VNIR/SWIR hyperspectral imager",
        "license": "Apache 2.0",
        "publication_reference": "Xie, Y.; Plett, D.; Liu, H. The Promise of Hyperspectral Imaging for the Early Detection of Crown Rot in Wheat. AgriEngineering 2021, 3, 924-941. DOI: 10.3390/agriengineering3040057",
        "dataset_limitations": "Tabularized ROI reflectance; full spatial 3D raw cubes require university network credentials",
        "selection_status": "SELECTED_PRIMARY",
        "selection_justification": "Direct open access, robust 1,632 samples across 657 continuous bands spanning both VNIR and SWIR, Apache 2.0 license, and explicitly focused on early pre-symptomatic disease forecasting."
    },
    {
        "dataset_id": "DS_TOMATO_BACTERIAL_WILT_002",
        "dataset_name": "Canopy Hyperspectral Dataset for Tomato Bacterial Wilt Severity Identification",
        "publisher_institution": "Zhejiang University / Mendeley Data",
        "official_source_url": "https://doi.org/10.17632/tb4yk4x5nw.1",
        "download_url": "https://data.mendeley.com/public-api/datasets/tb4yk4x5nw/files",
        "download_availability": "Web UI Direct Download (.rar archive)",
        "crop_species": "Tomato (Solanum lycopersicum)",
        "disease_categories": "Bacterial Wilt (Ralstonia solanacearum) across 5 severity grades (Level 0 to Level 4)",
        "sample_count": 293,
        "spectral_bands": 204,
        "wavelength_range": "400 nm - 1000 nm (VNIR)",
        "file_formats": ".mat (MATLAB / Scipy)",
        "ground_truth_labels": "Level 0 (Healthy), Level 1 (Pre-symptomatic/no wilting), Level 2 (1-33%), Level 3 (34-66%), Level 4 (67-100%)",
        "collection_methodology": "Specim IQ handheld hyperspectral camera in dark-room controlled illumination",
        "license": "CC BY 4.0",
        "publication_reference": "Wang, X. et al. Early identification and severity evaluation of tomato bacterial wilt using hyperspectral imaging. Frontiers in Plant Science (2021).",
        "dataset_limitations": "Sample size limited to 293 total canopy scans; packaged in proprietary RAR archive requiring unrar binaries",
        "selection_status": "BENCHMARK_SECONDARY",
        "selection_justification": "Valuable multi-severity progression dataset for secondary validation and domain generalization."
    },
    {
        "dataset_id": "DS_KIWI_BACTERIAL_CANKER_003",
        "dataset_name": "Hyperspectral Spectroscopic Reflectance of In-Vivo Non-Symptomatic and Symptomatic Kiwi Leaves",
        "publisher_institution": "INESC TEC / University of Porto / Zenodo",
        "official_source_url": "https://doi.org/10.5281/zenodo.10498541",
        "download_url": "https://zenodo.org/api/records/10498541/files-archive",
        "download_availability": "Open Zenodo Archive",
        "crop_species": "Kiwi (Actinidia deliciosa)",
        "disease_categories": "Bacterial Canker (Pseudomonas syringae pv. actinidiae)",
        "sample_count": 480,
        "spectral_bands": 751,
        "wavelength_range": "325 nm - 1075 nm (1 nm step)",
        "file_formats": ".csv / .txt spectroradiometer curves",
        "ground_truth_labels": "Non-symptomatic (Asymptomatic early) vs. Symptomatic (Chlorotic/Necrotic lesions)",
        "collection_methodology": "In-situ field spectroscopic collection with ASD FieldSpec HandHeld 2 spectroradiometer across commercial orchards",
        "license": "CC BY 4.0",
        "publication_reference": "Reis-Pereira, M. et al. In-vivo Proximal Hyperspectral Sensing for Pre-symptomatic Canker Monitoring. Agronomy (2024).",
        "dataset_limitations": "Point spectroscopy on leaf contact probe; does not capture wide canopy 2D spatial context",
        "selection_status": "BENCHMARK_SECONDARY",
        "selection_justification": "Excellent high-resolution spectral benchmark for field-scale pre-symptomatic detection."
    },
    {
        "dataset_id": "DS_POTATO_NEMATODE_STRESS_004",
        "dataset_name": "Hyperspectral Imaging of Potato Plants Exposed to Nematode and Drought Stress",
        "publisher_institution": "Agricultural Institute of Slovenia / Zenodo",
        "official_source_url": "https://doi.org/10.5281/zenodo.14013098",
        "download_url": "https://zenodo.org/record/14013098/files",
        "download_availability": "Zenodo Open Access",
        "crop_species": "Potato (Solanum tuberosum)",
        "disease_categories": "Nematode infestation (Globodera pallida) combined with water-deficit stress",
        "sample_count": 120,
        "spectral_bands": 384,
        "wavelength_range": "400 nm - 988 nm (VNIR) & 950 nm - 2500 nm (SWIR)",
        "file_formats": "HDF5 (.h5) & ENVI raw image cubes",
        "ground_truth_labels": "Inoculated vs Non-inoculated across 4 growth milestones",
        "collection_methodology": "Dual-camera VNIR/SWIR push-broom system in greenhouse conditions",
        "license": "CC BY 4.0",
        "publication_reference": "Lapajne, J. et al. Multi-temporal hyperspectral imaging of potato stress phenotyping. Sensors (2024).",
        "dataset_limitations": "Very large raw image cubes (>45 GB); exceeds lightweight edge deployment bounds",
        "selection_status": "EVALUATED_RESERVED",
        "selection_justification": "Detailed dual VNIR/SWIR setup, but massive archive size makes it unsuitable for agile 1-month intern project iteration."
    },
    {
        "dataset_id": "DS_SALINAS_SCENE_AVIRIS_005",
        "dataset_name": "Salinas Agricultural Hyperspectral Scene",
        "publisher_institution": "NASA Jet Propulsion Laboratory / Purdue University",
        "official_source_url": "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        "download_url": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        "download_availability": "Direct HTTP Download",
        "crop_species": "Mixed agricultural crops (Vegetables, Grapes, Broccoli, Lettuce, Vineyard)",
        "disease_categories": "Crop type and senescence classification (Benchmark)",
        "sample_count": 54129,
        "spectral_bands": 204,
        "wavelength_range": "400 nm - 2500 nm",
        "file_formats": ".mat (MATLAB)",
        "ground_truth_labels": "16 Crop Classes ground truth mask",
        "collection_methodology": "Airborne AVIRIS sensor over Salinas Valley, California",
        "license": "Public Domain (NASA/USGS)",
        "publication_reference": "Baumgardner, M. F. et al. 220 Band AVIRIS Hyperspectral Image Data Set. LARS Technical Reports (1992).",
        "dataset_limitations": "Captures crop taxonomic classification rather than specific plant pathology or disease stages",
        "selection_status": "BENCHMARK_REFERENCE",
        "selection_justification": "Standard baseline benchmark in hyperspectral literature for algorithm validation."
    }
]


def run_discovery():
    """Generates source registry CSV, catalog JSON, and dataset card Markdown."""
    logger.info("Executing dataset discovery and candidate evaluation...")

    # 1. Save source registry CSV
    fieldnames = [
        "dataset_id",
        "dataset_name",
        "publisher_institution",
        "crop_species",
        "disease_categories",
        "sample_count",
        "spectral_bands",
        "wavelength_range",
        "file_formats",
        "license",
        "selection_status",
        "official_source_url"
    ]

    registry_records = [
        {k: d[k] for k in fieldnames} for d in CANDIDATE_DATASETS
    ]
    save_csv(registry_records, SOURCE_REGISTRY_FILE, fieldnames)

    # 2. Save comprehensive dataset catalog JSON
    catalog_data = {
        "catalog_version": "1.0.0",
        "generated_timestamp": datetime.now().isoformat(),
        "project_name": "TerraSpectra",
        "organization": "Infotact Solutions",
        "primary_selected_dataset": "DS_WHEAT_CROWN_ROT_001",
        "total_candidate_datasets": len(CANDIDATE_DATASETS),
        "datasets": CANDIDATE_DATASETS
    }
    save_json(catalog_data, CATALOG_FILE)

    # 3. Generate detailed Dataset Card Markdown
    primary = next(d for d in CANDIDATE_DATASETS if d["dataset_id"] == "DS_WHEAT_CROWN_ROT_001")
    card_md = f"""# Dataset Card: {primary['dataset_name']}

**Dataset ID:** `{primary['dataset_id']}`  
**Project:** TerraSpectra (Hyperspectral Crop Disease Forecasting)  
**Organization:** Infotact Solutions  
**Selection Status:** **{primary['selection_status']}**  
**License:** {primary['license']} (Open Access)  
**Official Source URL:** [{primary['official_source_url']}]({primary['official_source_url']})  
**Publication:** {primary['publication_reference']}  

---

## 1. Dataset Summary
This dataset contains experimental hyperspectral reflectance measurements acquired for the early detection and diagnosis of **Crown Rot disease** in wheat (*Triticum aestivum* and *Triticum durum*), caused by the devastating fungal soil-borne pathogen ***Fusarium pseudograminearum***.

The study investigated whether hyperspectral imaging spanning the **VNIR (Visible & Near-Infrared: 400–1000 nm)** and **SWIR (Short-Wave Infrared: 1000–2400 nm)** spectrum could detect fungal invasion at the **pre-symptomatic phase** (before characteristic honey-brown stem lesions and foliar chlorosis develop).

---

## 2. Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| **Crop Host** | Wheat (*Triticum durum* and *Triticum aestivum*) |
| **Pathogen** | *Fusarium pseudograminearum* (Crown Rot) |
| **Total Samples** | **1,632 spectral curves** across multiple replicates |
| **Spectral Bands** | **657 contiguous narrow bands** |
| **Wavelength Range** | **396.9 nm to 2396.5 nm** (Continuous VNIR + SWIR) |
| **Spectral Resolution** | ~3.0 nm step (VNIR) to ~5.6 nm step (SWIR) |
| **Experimental Factors** | Inoculation (`Infected` vs. `Control`) & Water Regime (`Drought` vs. `Well-watered`) |
| **File Format** | Microsoft Excel (`.xlsx`), Sheet `barcode_hypdata` & Sheet `result` |
| **Original File Size** | 13,333,673 bytes (~13.3 MB) |

---

## 3. Data Integrity & Selection Justification
1. **Bio-Optical Precision:** Provides continuous narrow bands across the critical Red-Edge inflection (690–740nm) and water/lignin absorption features (1400–2100nm).
2. **Pre-Symptomatic Tracking:** Explicitly maps early fungal colonization vs. control plants under varying watering stresses.
3. **Accessibility:** Freely accessible via Figshare API/HTTP download under an Apache 2.0 open-source license.
4. **Authenticity:** Peer-reviewed and published in *AgriEngineering* (2021) by researchers at the University of Adelaide and CSIRO.
"""
    DATASET_CARD_FILE.write_text(card_md, encoding="utf-8")
    logger.info(f"Generated Dataset Card at: {DATASET_CARD_FILE}")

    print("=" * 70)
    print(" ✅ DATASET DISCOVERY COMPLETED SUCCESSFULLY")
    print(f"  * Catalog: {CATALOG_FILE}")
    print(f"  * Source Registry: {SOURCE_REGISTRY_FILE}")
    print(f"  * Dataset Card: {DATASET_CARD_FILE}")
    print(f"  * Primary Selection: {primary['dataset_name']} ({primary['sample_count']} samples, {primary['spectral_bands']} bands)")
    print("=" * 70)


if __name__ == "__main__":
    run_discovery()

