# 🛰️ 01. Dataset Collection Module

**Project:** TerraSpectra – Hyperspectral Crop Disease Forecasting  
**Organization:** Infotact Solutions  
**Domain:** Data Science & Machine Learning (Agricultural Remote Sensing)  
**Lead:** Vishal Yadav (`vyadav276203@gmail.com`)  

---

## 1. Module Overview
The `01_Dataset_Collection` module implements an enterprise-grade, reproducible data acquisition, validation, and cataloging pipeline for agricultural hyperspectral imagery and narrow-band reflectance signatures. 

Unlike conventional computer vision tasks that rely on RGB photography, hyperspectral disease forecasting requires authentic narrow-band spectral measurements capturing the biochemical and cellular changes associated with pathogen colonization before visual foliar symptoms emerge.

---

## 2. Purpose of Dataset Collection
1. **Source Verifiable Scientific Data:** Acquire authentic, peer-reviewed hyperspectral datasets from certified open repositories (Figshare, Zenodo, Mendeley Data).
2. **Ensure Data Provenance & Traceability:** Maintain cryptographic SHA-256 checksums, automated manifests, and collection metadata.
3. **Automate Pipeline Execution:** Provide clean CLI scripts for dataset discovery, resilient downloading (with retry and size validation), format inspection, and inventory reporting.
4. **Enforce Engineering Integrity:** Guard against synthetic or fabricated data, ensuring that only authentic spectral measurements feed into downstream preprocessing and modeling.

---

## 3. Hyperspectral Imaging Basics in Plant Pathology
- **Visible Spectrum (VIS, 400–700 nm):** Dominated by chlorophyll $a$ and $b$ absorption (blue ~450nm and red ~670nm) and carotenoid/xanthophyll pigments.
- **Red-Edge Transition Zone (690–740 nm):** The steep slope between red chlorophyll absorption and Near-Infrared scattering. When a crop is infected, loss of chlorophyll causes the red-edge derivative peak to shift toward shorter wavelengths ("blue-shift").
- **Near-Infrared (NIR, 700–1000 nm):** High reflectance plateau driven by multiple internal light scatterings within the spongy mesophyll cell structure. Fungal hyphae colonizing intercellular spaces disrupt cell walls, causing significant NIR reflectance drops.
- **Short-Wave Infrared (SWIR, 1000–2500 nm):** Controlled by plant water content (absorption troughs at ~1450nm and ~1940nm), protein, starch, and structural lignin.

---

## 4. Evaluated Dataset Sources & Candidates

The pipeline evaluated 5 authentic candidates cataloged in `dataset_info/source_registry.csv`:

| Dataset ID | Crop Species | Pathogen / Condition | Bands | Wavelength Range | Source Repository | Selection Status |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| **DS_WHEAT_CROWN_ROT_001** | **Wheat** | **Crown Rot (*Fusarium pseudograminearum*)** | **657** | **400 – 2400 nm** | **Figshare (Univ. of Adelaide)** | **🏆 PRIMARY SELECTED** |
| `DS_TOMATO_BACTERIAL_WILT_002` | Tomato | Bacterial Wilt (*Ralstonia solanacearum*) | 204 | 400 – 1000 nm | Mendeley Data | Secondary Benchmark |
| `DS_KIWI_BACTERIAL_CANKER_003` | Kiwi | Bacterial Canker (*Psa*) | 751 | 325 – 1075 nm | Zenodo (INESC TEC) | Secondary Benchmark |
| `DS_POTATO_NEMATODE_STRESS_004` | Potato | Nematodes (*Globodera*) + Drought | 384 | 400 – 2500 nm | Zenodo (AIS) | Evaluated / Reserved |
| `DS_SALINAS_SCENE_AVIRIS_005` | Mixed Crops | Crop Taxonomy Benchmark | 204 | 400 – 2500 nm | NASA AVIRIS / Purdue | Reference Benchmark |

---

## 5. Selected Dataset Justification
**Wheat Crown Rot Hyperspectral Dataset** (`DS_WHEAT_CROWN_ROT_001`) was selected as the primary dataset because:
1. **Direct Focus on Early Disease Detection:** The experimental design tracks *Fusarium pseudograminearum* fungal colonization across inoculated vs. control wheat lines under varied watering regimes.
2. **Continuous Ultra-High Spectral Resolution:** 657 contiguous narrow bands covering both VNIR (400–1000nm) and SWIR (1000–2400nm) at ~3–5nm intervals.
3. **Permissive Open Licensing:** Licensed under **Apache 2.0**, allowing full academic and production utilization with proper attribution.
4. **Verifiable Peer-Reviewed Publication:** Published in *AgriEngineering* (2021) by researchers at the University of Adelaide and CSIRO.

---

## 6. Dataset Specifications (`DS_WHEAT_CROWN_ROT_001`)
- **Total Samples:** 1,632 spectral curves
- **Total Spectral Bands:** 657 narrow bands
- **Wavelength Bounds:** 396.9 nm to 2396.5 nm
- **File Format:** Microsoft Excel OpenXML Spreadsheet (`.xlsx`)
- **Size:** 13,333,673 bytes (~13.3 MB)
- **Experimental Target Labels:**
  - Disease Status: `Infected` (inoculated) vs. `Control` (mock-inoculated)
  - Watering Status: `Drought` vs. `Well-watered`
  - Genotypes: Multiple bread and durum wheat cultivars (*Triticum durum*, *Triticum aestivum*)

---

## 7. Folder Structure

```text
01_Dataset_Collection/
│
├── raw_data/                           # Immutable raw data store
│   ├── hyperspectral/                  # Downloaded raw datasets (data_AgEng_review_paper.xlsx)
│   ├── metadata/                       # Sensor and platform specification sheets
│   └── README.md                       # Raw data storage policies
│
├── processed_data/                     # Standardized tabular/matrix spectral exports
│
├── dataset_info/                       # Provenance, catalog & inventory reports
│   ├── dataset_catalog.json            # Machine-readable catalog of all investigated datasets
│   ├── source_registry.csv             # Structured CSV registry of candidate sources
│   ├── dataset_manifest.csv            # File-level manifest with SHA-256 and timestamps
│   ├── dataset_card.md                 # Detailed research dataset card
│   ├── collection_report.json          # Download execution log and telemetry
│   ├── checksums_sha256.json           # Cryptographic integrity hashes
│   ├── dataset_inventory.json          # Statistical data inventory output
│   ├── dataset_inventory_report.md     # Markdown inventory summary report
│   └── outputs/                        # Inventory charts and diagnostic figures
│       └── 01_dataset_inventory_summary.png
│
├── scripts/                            # Modular pipeline execution scripts
│   ├── discover_datasets.py            # Generates source registry and dataset catalog
│   ├── download_dataset.py             # Resilient downloader with retry and size checks
│   ├── extract_archives.py             # Safe archive extraction (Zip Slip protection)
│   ├── validate_dataset.py             # File-level and hyperspectral domain validator
│   ├── generate_manifest.py            # Generates dataset_manifest.csv
│   └── generate_checksums.py           # Computes and verifies SHA-256 checksums
│
├── utils/                              # Reusable engineering utilities
│   ├── file_utils.py                   # Chunked downloading and archive extraction
│   ├── metadata_utils.py               # SHA-256 hashing and JSON/CSV serialization
│   └── validation_utils.py             # Format-specific spectral data validation
│
├── logs/                               # Execution logs
│   └── dataset_collection.log          # Detailed execution traces and timestamps
│
├── tests/                              # Automated test suite (pytest)
│   ├── conftest.py                     # Synthetic isolated test fixtures
│   ├── test_file_validation.py         # File existence and format checks
│   ├── test_metadata.py                # Checksum and serialization tests
│   ├── test_manifest.py                # Catalog and manifest structure tests
│   └── test_download_pipeline.py       # Download retry, skip and extraction tests
│
├── data_analysis.py                    # Dataset inventory and collection quality assessment
├── config.py                           # Central configuration module
├── requirements.txt                    # Module dependency specification
├── .gitignore                          # Module-specific ignore rules
└── README.md                           # This documentation
```

---

## 8. Installation & Environment Setup

Ensure you are in the project root with the active Python virtual environment:

```powershell
# Windows PowerShell
Set-Location "c:\Users\Vishal Yadav\OneDrive\Desktop\TerraSpectra"

# Verify Python
python --version

# Install dependencies
python -m pip install -r 01_Dataset_Collection\requirements.txt
```

---

## 9. Configuration (`config.py`)
Key parameters can be adjusted in `01_Dataset_Collection/config.py`:
- `DOWNLOAD_TIMEOUT_SECONDS`: Maximum HTTP request wait time (default: 60s).
- `MAX_RETRIES`: Number of retry attempts on network error (default: 3).
- `RETRY_BACKOFF_FACTOR`: Exponential backoff multiplier (default: 2.0).
- `PRIMARY_DATASET_URL`: Direct endpoint for the Figshare dataset.
- `PRIMARY_DATASET_EXPECTED_SIZE`: Verified byte size (13,333,673 bytes).

---

## 10. Step-by-Step Execution Workflow (Windows PowerShell)

Run each step sequentially from the project root:

### Step 1: Discover and Catalog Datasets
Generates `source_registry.csv`, `dataset_catalog.json`, and `dataset_card.md`:
```powershell
python 01_Dataset_Collection\scripts\discover_datasets.py
```

### Step 2: Download the Primary Dataset
Downloads `data_AgEng_review_paper.xlsx` directly into `raw_data/hyperspectral/`:
```powershell
python 01_Dataset_Collection\scripts\download_dataset.py
```

### Step 3: Extract Archives (If Applicable)
Safely unpacks any archives while guarding against path traversal:
```powershell
python 01_Dataset_Collection\scripts\extract_archives.py
```

### Step 4: Validate Dataset Integrity & Hyperspectral Structure
Verifies file readability, band counts, wavelength boundaries, and class labels:
```powershell
python 01_Dataset_Collection\scripts\validate_dataset.py
```

### Step 5: Generate Dataset Manifest
Catalogs every tracked file with timestamps, file size, and SHA-256 hash:
```powershell
python 01_Dataset_Collection\scripts\generate_manifest.py
```

### Step 6: Generate & Verify SHA-256 Checksums
Creates `checksums_sha256.json` and verifies existing files:
```powershell
# Generate hashes
python 01_Dataset_Collection\scripts\generate_checksums.py

# Verify integrity
python 01_Dataset_Collection\scripts\generate_checksums.py --verify
```

### Step 7: Run Data Inventory & Quality Analysis
Generates sample distributions, class counts, and visual assessment chart:
```powershell
python 01_Dataset_Collection\data_analysis.py
```

### Step 8: Run Automated Unit Tests
Executes the isolated test suite using `pytest`:
```powershell
python -m pytest 01_Dataset_Collection\tests -v
```

---

## 11. Expected Outputs

Upon running the complete pipeline, the following artifacts are generated:
- `raw_data/hyperspectral/data_AgEng_review_paper.xlsx`: 13.3 MB original dataset file.
- `dataset_info/source_registry.csv`: Catalog of all investigated candidates.
- `dataset_info/dataset_catalog.json`: Structured metadata on candidates and primary choice.
- `dataset_info/dataset_card.md`: Research paper details, host species, and pathology context.
- `dataset_info/collection_report.json`: Download timestamp, network duration, and size audit.
- `dataset_info/validation_report.json`: Spectral band counts, wavelength ranges, and null checks.
- `dataset_info/dataset_manifest.csv`: Traceable file inventory table.
- `dataset_info/checksums_sha256.json`: Checksum registry.
- `dataset_info/dataset_inventory.json`: Dataset inventory metrics.
- `dataset_info/dataset_inventory_report.md`: Markdown report of data inventory.
- `dataset_info/outputs/01_dataset_inventory_summary.png`: Visual bar charts and wavelength histogram.
- `logs/dataset_collection.log`: Complete timestamped execution trace.

---

## 12. Troubleshooting & FAQ

- **Network Connection Error / Proxy Failure:**  
  *Fix:* Check internet connectivity or company firewall. `download_dataset.py` includes a 3-retry exponential backoff mechanism.
- **Excel Reading Permission / Dependency Warning:**  
  *Fix:* Ensure `openpyxl` is installed (`python -m pip install openpyxl`).
- **File Size Mismatch:**  
  *Fix:* If a partial download occurs, run `python 01_Dataset_Collection\scripts\download_dataset.py --force` to re-download.

---

## 13. Limitations & Downstream Readiness
- **Spatial Resolution:** The dataset provides Region of Interest (ROI) average spectral curves across whole plants rather than raw 3D spatial cubes. This is ideal for 1D spectral modeling, narrow-band index calculation, and rapid machine learning prototyping.
- **Next Module Readiness:** The dataset is fully validated, completely free of missing labels, covers both VNIR and SWIR (400–2400nm), and is **100% ready for `02_Data_Preprocessing`**.

---

## 14. References & Attribution

1. **Xie, Y.; Plett, D.; Liu, H.** (2021). *The Promise of Hyperspectral Imaging for the Early Detection of Crown Rot in Wheat.* **AgriEngineering**, 3(4), 924–941. DOI: [10.3390/agriengineering3040057](https://doi.org/10.3390/agriengineering3040057)
2. **Liu, Huajian** (2021). *Pre-processed hyperspectral data for crown rot detection in wheat.* **Figshare Repository**. DOI: [10.25909/17075138](https://doi.org/10.25909/17075138)
