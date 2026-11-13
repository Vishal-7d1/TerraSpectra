"""
File handling utilities: downloading with retry, resume, chunking, and safe extraction.
"""

from pathlib import Path
import time
import zipfile
import tarfile
from typing import Optional, Callable
import urllib.request
import urllib.error

from config import (
    DOWNLOAD_TIMEOUT_SECONDS,
    DOWNLOAD_CHUNK_SIZE_BYTES,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    setup_logger,
)

logger = setup_logger("file_utils")


def download_file(
    url: str,
    destination: Path,
    expected_size: Optional[int] = None,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bool:
    """
    Downloads a file over HTTP/HTTPS with retry logic, chunking, and size verification.

    Parameters:
        url (str): Source URL.
        destination (Path): Target file destination path.
        expected_size (Optional[int]): Expected size in bytes for verification.
        overwrite (bool): If False and destination exists, download is skipped.
        progress_callback (Optional[Callable[[int, int], None]]): Callback(downloaded_bytes, total_bytes).

    Returns:
        bool: True if download/file is ready and verified, False otherwise.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        if expected_size and destination.stat().st_size == expected_size:
            logger.info(f"File already exists with expected size ({destination.stat().st_size} bytes): {destination.name}")
            return True
        elif not expected_size and destination.stat().st_size > 0:
            logger.info(f"File already exists: {destination.name}")
            return True

    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    headers = {
        "User-Agent": "TerraSpectra-DataCollector/1.0 (Infotact Solutions Internship Project; Agricultural HSI)"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Downloading from {url} (Attempt {attempt}/{MAX_RETRIES})...")
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                total_size = response.headers.get("Content-Length")
                total_size = int(total_size) if total_size else expected_size or 0

                downloaded = 0
                with open(temp_path, "wb") as f_out:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE_BYTES)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size:
                            progress_callback(downloaded, total_size)

            actual_size = temp_path.stat().st_size
            if expected_size and actual_size != expected_size:
                raise ValueError(
                    f"Size mismatch: expected {expected_size} bytes, got {actual_size} bytes."
                )

            # Atomic move
            if destination.exists():
                destination.unlink()
            temp_path.rename(destination)
            logger.info(f"Successfully downloaded {destination.name} ({actual_size:,} bytes).")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            logger.warning(f"Download failed on attempt {attempt}/{MAX_RETRIES}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to download from {url} after {MAX_RETRIES} attempts.")
                return False

    return False


def safe_extract_archive(archive_path: Path, extract_to: Path) -> list[Path]:
    """
    Safely extracts .zip or .tar/.tar.gz archives preventing path traversal attacks (Zip Slip).

    Parameters:
        archive_path (Path): Path to archive.
        extract_to (Path): Destination directory.

    Returns:
        list[Path]: List of extracted file paths.
    """
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    extracted_files = []

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                target_path = (extract_to / member.filename).resolve()
                if not str(target_path).startswith(str(extract_to.resolve())):
                    raise ValueError(f"Path traversal detected in archive: {member.filename}")
            zf.extractall(extract_to)
            for member in zf.infolist():
                extracted_files.append(extract_to / member.filename)

    elif archive_path.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                target_path = (extract_to / member.name).resolve()
                if not str(target_path).startswith(str(extract_to.resolve())):
                    raise ValueError(f"Path traversal detected in archive: {member.name}")
            tf.extractall(extract_to)
            for member in tf.getmembers():
                extracted_files.append(extract_to / member.name)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    logger.info(f"Safely extracted {len(extracted_files)} files to {extract_to}")
    return extracted_files

