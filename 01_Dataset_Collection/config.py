"""
Configuration module for 01_Dataset_Collection.
Centralizes paths, network settings, download retry policies, and dataset registries.
"""

from pathlib import Path
import logging

# Base directories
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

RAW_DATA_DIR = MODULE_DIR / "raw_data"
HYPERSPECTRAL_RAW_DIR = RAW_DATA_DIR / "hyperspectral"
METADATA_RAW_DIR = RAW_DATA_DIR / "metadata"

PROCESSED_DATA_DIR = MODULE_DIR / "processed_data"
DATASET_INFO_DIR = MODULE_DIR / "dataset_info"
DATASET_INFO_OUTPUTS_DIR = DATASET_INFO_DIR / "outputs"
LOGS_DIR = MODULE_DIR / "logs"

# Ensure essential directories exist
for directory in [
    RAW_DATA_DIR,
    HYPERSPECTRAL_RAW_DIR,
    METADATA_RAW_DIR,
    PROCESSED_DATA_DIR,
    DATASET_INFO_DIR,
    DATASET_INFO_OUTPUTS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
CATALOG_FILE = DATASET_INFO_DIR / "dataset_catalog.json"
MANIFEST_FILE = DATASET_INFO_DIR / "dataset_manifest.csv"
SOURCE_REGISTRY_FILE = DATASET_INFO_DIR / "source_registry.csv"
DATASET_CARD_FILE = DATASET_INFO_DIR / "dataset_card.md"
COLLECTION_REPORT_FILE = DATASET_INFO_DIR / "collection_report.json"
CHECKSUM_FILE = DATASET_INFO_DIR / "checksums_sha256.json"

# Logging configuration
LOG_FILE = LOGS_DIR / "dataset_collection.log"

def setup_logger(name: str = "dataset_collection") -> logging.Logger:
    """Configures and returns a logger instance writing to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

# Network & Download policies
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB chunk
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0

# Primary dataset download settings
PRIMARY_DATASET_ID = "DS_WHEAT_CROWN_ROT_001"
PRIMARY_DATASET_FILENAME = "data_AgEng_review_paper.xlsx"
PRIMARY_DATASET_URL = "https://ndownloader.figshare.com/files/31569671"
PRIMARY_DATASET_EXPECTED_SIZE = 13333673

