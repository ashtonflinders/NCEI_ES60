"""
nav_checker.py

Author:
    Ashton Flinders
    andrealphus@gmail.com

Date Created:
    2026-04-09

Description:
    This script organizes and classifies Simrad ES60 *.raw echosounder files
    based on the presence of usable navigation data. It is designed to assist
    in preprocessing datasets for acoustic analysis using the echopype library.

    The script performs the following tasks within a user-specified source
    directory:

        1. Creates or uses five classification subdirectories:
            - raw_w_nav   : *.raw files containing usable navigation data.
            - raw_no_nav  : *.raw files lacking usable navigation data.
            - raw_error   : *.raw files that could not be opened or processed.
            - calibration : Directories whose names suggest calibration content.
            - other       : All non-*.raw files.

        2. Handles all top-level non-*.raw files by moving or copying them
           to the "other" directory.

        3. Handles any top-level subdirectories with names containing
           "calibration" or exactly equal to "cal" (case-insensitive) by
           moving or copying them to the "calibration" directory.

        4. Processes only the remaining top-level *.raw files using echopype
           to determine whether usable navigation data are present.

        5. Moves or copies *.raw files to the appropriate classification
           directory based on the navigation-data classification.

        6. Generates a summary table (raw_nav_summary.csv) in the destination
           directory containing diagnostic information for each file.

Notes:
    - Although this script specifies sonar_model="EK60" when opening files
      with echopype, it is intended for processing Simrad ES60 data. The
      ES60 and EK60 echosounders share the same .raw file format, and
      echopype uses the "EK60" designation to support both systems.

Usage:
    Run the script from the command line by providing the path to the
    source directory containing the data:

        python nav_checker.py /path/to/source_directory

    Or run interactively and enter the directory when prompted:

        python nav_checker.py

Configuration:
    READ_ONLY_MODE:
        Controls whether the script modifies the filesystem.

        - True  : Analysis-only mode. The script inspects files and prints
                  the actions it would take, but does NOT create directories,
                  move files, or copy files. A summary CSV is written only if
                  the destination directory already exists.
        - False : The script performs filesystem operations as defined by
                  MOVE_FILES.

    MOVE_FILES:
        Determines how files and directories are handled when
        READ_ONLY_MODE is False.

        - True  : Files and directories are moved into classification
                  folders inside the source directory. Originals are removed
                  from their original locations.
        - False : Files and directories are copied into classification
                  folders inside OUTPUT_DIR. Originals remain in place.
                  OUTPUT_DIR must be specified.

    OUTPUT_DIR:
        Destination directory used when MOVE_FILES is False. This keeps the
        source directory unchanged while writing organized copies elsewhere.

        - None  : Invalid when MOVE_FILES is False.
        - Path  : Destination directory for copied outputs.

    CREATE_OUTPUT_DIR:
        Controls behavior when MOVE_FILES is False and OUTPUT_DIR does not
        already exist.

        - True  : Create OUTPUT_DIR automatically.
        - False : Raise FileNotFoundError if OUTPUT_DIR does not exist.

    USE_SENTENCE_FOR_NAV:
        If True, sentence_type content can be used as part of the decision
        that a file has usable navigation.
        If False, only valid latitude and longitude are used.

Dependencies:
    - Python 3.10+
    - echopype
    - numpy
    - pandas
"""

from pathlib import Path
import os
import shutil
import sys

import echopype as ep
import numpy as np
import pandas as pd


READ_ONLY_MODE = True
MOVE_FILES = False
OUTPUT_DIR = None
CREATE_OUTPUT_DIR = True
USE_SENTENCE_FOR_NAV = True


def relpath(path: Path) -> str:
    """
    Return a path relative to the current working directory for cleaner
    printing. Falls back to os.path.relpath if necessary.
    """
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return os.path.relpath(str(path), start=str(Path.cwd()))


def has_any_valid_numeric(values) -> bool:
    """Return True if array-like contains any finite numeric values."""
    if values is None:
        return False

    arr = np.asarray(values)

    if arr.size == 0:
        return False

    try:
        arr_float = arr.astype(float)
        return bool(np.any(np.isfinite(arr_float)))
    except (TypeError, ValueError):
        return False


def count_valid_numeric(values) -> int:
    """Return count of finite numeric values in an array-like object."""
    if values is None:
        return 0

    arr = np.asarray(values)

    if arr.size == 0:
        return 0

    try:
        arr_float = arr.astype(float)
        return int(np.sum(np.isfinite(arr_float)))
    except (TypeError, ValueError):
        return 0


def has_any_valid_sentence(values) -> bool:
    """Return True if array-like contains any non-empty, non-NaN entries."""
    if values is None:
        return False

    arr = np.asarray(values)

    if arr.size == 0:
        return False

    for val in arr.ravel():
        if val is None:
            continue

        if isinstance(val, bytes):
            if val.strip():
                return True
            continue

        if isinstance(val, str):
            stripped = val.strip()
            if stripped and stripped.lower() != "nan":
                return True
            continue

        try:
            if not np.isnan(val):
                return True
        except TypeError:
            if str(val).strip():
                return True

    return False


def classify_raw_file(raw_file: Path) -> dict:
    """Open a raw file and classify whether it contains usable nav data."""
    result = {
        "file": raw_file.name,
        "source_path": str(raw_file),
        "usable_nav": False,
        "nav_decision_basis": "",
        "has_lat": False,
        "has_lon": False,
        "has_latlon": False,
        "lat_non_nan_count": 0,
        "lon_non_nan_count": 0,
        "has_sentence": False,
        "n_time1": 0,
        "use_sentence_for_nav": USE_SENTENCE_FOR_NAV,
        "error": "",
    }

    try:
        ed = ep.open_raw(str(raw_file), sonar_model="EK60")
        platform = ed["Platform"]

        lat = (
            platform["latitude"].values
            if "latitude" in platform.data_vars
            else None
        )
        lon = (
            platform["longitude"].values
            if "longitude" in platform.data_vars
            else None
        )
        sentence = (
            platform["sentence_type"].values
            if "sentence_type" in platform.data_vars
            else None
        )

        has_lat = has_any_valid_numeric(lat)
        has_lon = has_any_valid_numeric(lon)
        has_latlon = has_lat and has_lon

        lat_non_nan_count = count_valid_numeric(lat)
        lon_non_nan_count = count_valid_numeric(lon)

        has_sentence = has_any_valid_sentence(sentence)
        n_time1 = int(platform.sizes.get("time1", 0))

        if USE_SENTENCE_FOR_NAV:
            usable_nav = has_latlon or has_sentence
            if has_latlon and has_sentence:
                nav_decision_basis = "latlon+sentence"
            elif has_latlon:
                nav_decision_basis = "latlon"
            elif has_sentence:
                nav_decision_basis = "sentence"
            else:
                nav_decision_basis = "none"
        else:
            usable_nav = has_latlon
            nav_decision_basis = "latlon" if has_latlon else "none"

        result.update(
            {
                "usable_nav": usable_nav,
                "nav_decision_basis": nav_decision_basis,
                "has_lat": has_lat,
                "has_lon": has_lon,
                "has_latlon": has_latlon,
                "lat_non_nan_count": lat_non_nan_count,
                "lon_non_nan_count": lon_non_nan_count,
                "has_sentence": has_sentence,
                "n_time1": n_time1,
            }
        )

    except Exception as exc:
        result["error"] = str(exc)

    return result


def transfer_path(source: Path, destination: Path) -> None:
    """Copy or move a file or directory depending on global settings."""
    if READ_ONLY_MODE:
        print(f"READ ONLY: {relpath(source)} -> {relpath(destination)}")
        return

    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {relpath(destination)}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if MOVE_FILES:
        shutil.move(str(source), str(destination))
    else:
        if source.is_dir():
            shutil.copytree(str(source), str(destination))
        else:
            shutil.copy2(str(source), str(destination))


def is_calibration_dir(path: Path) -> bool:
    """Return True if directory name suggests calibration content."""
    name = path.name.lower()
    return ("calibration" in name) or (name == "cal")


def prepare_destination_base_dir(base_dir: Path) -> Path:
    """Determine and validate the destination base directory."""
    if not MOVE_FILES and OUTPUT_DIR is None:
        raise ValueError(
            "OUTPUT_DIR must be specified when MOVE_FILES=False."
        )

    if MOVE_FILES and OUTPUT_DIR is not None:
        raise ValueError(
            "OUTPUT_DIR should be None when MOVE_FILES=True."
        )

    if MOVE_FILES:
        return base_dir

    destination_base_dir = Path(OUTPUT_DIR).expanduser().resolve()

    if destination_base_dir.exists():
        if not destination_base_dir.is_dir():
            raise NotADirectoryError(
                f"Output path exists but is not a directory: "
                f"{relpath(destination_base_dir)}"
            )

        return destination_base_dir

    if CREATE_OUTPUT_DIR:
        if READ_ONLY_MODE:
            print(
                f"READ ONLY: Would create output directory: "
                f"{relpath(destination_base_dir)}"
            )
        else:
            destination_base_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Created output directory: "
                f"{relpath(destination_base_dir)}"
            )

        return destination_base_dir

    raise FileNotFoundError(
        f"Output directory does not exist: {relpath(destination_base_dir)}"
    )


def create_category_dirs(destination_base_dir: Path) -> None:
    """Create output category directories if filesystem changes are enabled."""
    category_dirs = [
        destination_base_dir / "raw_w_nav",
        destination_base_dir / "raw_no_nav",
        destination_base_dir / "raw_error",
        destination_base_dir / "calibration",
        destination_base_dir / "other",
    ]

    if READ_ONLY_MODE:
        for category_dir in category_dirs:
            print(f"READ ONLY: Would create directory: {relpath(category_dir)}")
        return

    for category_dir in category_dirs:
        category_dir.mkdir(parents=True, exist_ok=True)


def organize_non_raw_and_calibration_dirs(
    base_dir: Path,
    destination_base_dir: Path,
) -> None:
    """
    Organize top-level non-.raw files and calibration-like top-level
    directories into the destination directory.
    """
    other_dir = destination_base_dir / "other"
    calibration_dir = destination_base_dir / "calibration"

    protected_names = {
        "raw_w_nav",
        "raw_no_nav",
        "raw_error",
        "calibration",
        "other",
        "raw_nav_summary.csv",
    }

    for item in sorted(base_dir.iterdir()):
        if item.name in protected_names:
            continue

        if item.resolve() == destination_base_dir.resolve():
            continue

        if item.is_file() and item.suffix.lower() != ".raw":
            destination = other_dir / item.name
            transfer_path(item, destination)
            print(f"{relpath(item)} -> {relpath(destination)}")

        elif item.is_dir() and is_calibration_dir(item):
            destination = calibration_dir / item.name
            transfer_path(item, destination)
            print(f"{relpath(item)} -> {relpath(destination)}")


def process_raw_files(
    base_dir: Path,
    destination_base_dir: Path,
) -> pd.DataFrame:
    """Classify remaining top-level raw files and organize them by nav status."""
    raw_w_nav_dir = destination_base_dir / "raw_w_nav"
    raw_no_nav_dir = destination_base_dir / "raw_no_nav"
    raw_error_dir = destination_base_dir / "raw_error"

    raw_files = sorted(
        [
            path for path in base_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".raw"
        ]
    )

    results = []

    for raw_file in raw_files:
        result = classify_raw_file(raw_file)

        if result["error"]:
            destination = raw_error_dir / raw_file.name
            category = "raw_error"
        elif result["usable_nav"]:
            destination = raw_w_nav_dir / raw_file.name
            category = "raw_w_nav"
        else:
            destination = raw_no_nav_dir / raw_file.name
            category = "raw_no_nav"

        result["destination_path"] = str(destination)
        result["category"] = category
        results.append(result)

        transfer_path(raw_file, destination)

        print(
            f"{relpath(raw_file)} -> {relpath(destination)} | "
            f"category={category} | "
            f"usable_nav={result['usable_nav']} | "
            f"basis={result['nav_decision_basis']} | "
            f"has_lat={result['has_lat']} | "
            f"has_lon={result['has_lon']} | "
            f"lat_count={result['lat_non_nan_count']} | "
            f"lon_count={result['lon_non_nan_count']} | "
            f"has_sentence={result['has_sentence']} | "
            f"use_sentence_for_nav={result['use_sentence_for_nav']} | "
            f"n_time1={result['n_time1']} | "
            f"error={result['error']}"
        )

    return pd.DataFrame(results)


def main(source_dir: str) -> pd.DataFrame:
    """Organize source directory contents and classify raw files by nav status."""
    base_dir = Path(source_dir).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {relpath(base_dir)}"
        )

    if not base_dir.is_dir():
        raise NotADirectoryError(
            f"Source path is not a directory: {relpath(base_dir)}"
        )

    destination_base_dir = prepare_destination_base_dir(base_dir)

    if READ_ONLY_MODE:
        print(
            "READ ONLY MODE: No directories will be created and no files "
            "or folders will be moved or copied."
        )

    print(f"Source directory: {relpath(base_dir)}")
    print(f"Destination directory: {relpath(destination_base_dir)}")
    print(f"MOVE_FILES: {MOVE_FILES}")
    print(f"CREATE_OUTPUT_DIR: {CREATE_OUTPUT_DIR}")

    create_category_dirs(destination_base_dir)

    organize_non_raw_and_calibration_dirs(
        base_dir=base_dir,
        destination_base_dir=destination_base_dir,
    )

    df = process_raw_files(
        base_dir=base_dir,
        destination_base_dir=destination_base_dir,
    )

    summary_path = destination_base_dir / "raw_nav_summary.csv"

    print("\nSummary counts:")
    print(f"Source directory: {relpath(base_dir)}")
    print(f"Destination directory: {relpath(destination_base_dir)}")
    print(f"Total raw files processed: {len(df)}")
    print(f"Usable nav: {(df['usable_nav'] == True).sum() if not df.empty else 0}")
    print(
        f"No nav: "
        f"{((df['usable_nav'] == False) & (df['error'] == '')).sum() if not df.empty else 0}"
    )
    print(f"Errors: {(df['error'] != '').sum() if not df.empty else 0}")

    if READ_ONLY_MODE and not destination_base_dir.exists():
        print(
            f"READ ONLY: Summary not written because destination directory "
            f"does not exist: {relpath(destination_base_dir)}"
        )
    else:
        if not READ_ONLY_MODE:
            summary_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(summary_path, index=False)
        print(f"Summary written to: {relpath(summary_path)}")

    return df


if __name__ == "__main__":
    if len(sys.argv) == 2:
        source_dir = sys.argv[1]
    elif len(sys.argv) == 1:
        try:
            source_dir = input(
                "Enter the path to the source directory containing the data: "
            ).strip()
            if not source_dir:
                raise ValueError("No directory provided.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            sys.exit(1)
    else:
        print(f"Usage: python {Path(sys.argv[0]).name} /path/to/source_directory")
        sys.exit(1)

    try:
        summary_df = main(source_dir)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("\nSummary table:")
    print(summary_df)