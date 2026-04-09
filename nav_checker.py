"""
nav_checker.py

Author:
    Ashton Flinders
    U.S. Geological Survey (USGS)
    Email: aflinders@usgs.gov

Date Created:
    2026-04-09

Description:
    This script organizes and classifies Simrad ES60 *.raw echosounder files
    based on the presence of usable navigation data. It is designed to assist
    in preprocessing datasets for acoustic analysis using the echopype library.

    The script performs the following tasks within a user-specified source
    directory:

        1. Creates five subdirectories:
            - raw_w_nav   : *.raw files containing usable navigation data.
            - raw_no_nav  : *.raw files lacking usable navigation data.
            - raw_error   : *.raw files that could not be opened or processed.
            - calibration : Directories whose names suggest calibration content.
            - other       : All non-*.raw files.

        2. Moves all top-level non-*.raw files to the "other" directory.

        3. Moves any top-level subdirectories with names containing
           "calibration" or exactly equal to "cal" (case-insensitive)
           to the "calibration" directory.

        4. Processes only the remaining top-level *.raw files using echopype
           to determine whether usable navigation data are present.

        5. Moves *.raw files to the appropriate directory based on the
           classification.

        6. Generates a summary table (raw_nav_summary.csv) in the top-level
           source directory containing diagnostic information for each file.

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
                  move files, or copy files. A summary CSV is still written.
        - False : The script performs filesystem operations as defined by
                  MOVE_FILES.

    MOVE_FILES:
        Determines how files and directories are handled when
        READ_ONLY_MODE is False.

        - False : Files and directories are copied to their destination
                  folders. Originals remain in place.
        - True  : Files and directories are moved to their destination
                  folders, removing them from their original location.

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


def organize_non_raw_and_calibration_dirs(base_dir: Path) -> None:
    """
    Move top-level non-.raw files to 'other' and calibration-like top-level
    directories to 'calibration'.
    """
    other_dir = base_dir / "other"
    calibration_dir = base_dir / "calibration"

    for item in sorted(base_dir.iterdir()):
        if item.name in {
            "raw_w_nav",
            "raw_no_nav",
            "raw_error",
            "calibration",
            "other",
            "raw_nav_summary.csv",
        }:
            continue

        if item.is_file() and item.suffix.lower() != ".raw":
            destination = other_dir / item.name
            transfer_path(item, destination)
            print(f"{relpath(item)} -> other")

        elif item.is_dir() and is_calibration_dir(item):
            destination = calibration_dir / item.name
            transfer_path(item, destination)
            print(f"{relpath(item)} -> calibration")


def process_raw_files(base_dir: Path) -> pd.DataFrame:
    """Classify remaining top-level raw files and organize them by nav status."""
    raw_w_nav_dir = base_dir / "raw_w_nav"
    raw_no_nav_dir = base_dir / "raw_no_nav"
    raw_error_dir = base_dir / "raw_error"

    raw_files = sorted(
        [
            path for path in base_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".raw"
        ]
    )

    results = []

    for raw_file in raw_files:
        result = classify_raw_file(raw_file)
        results.append(result)

        if result["error"]:
            destination = raw_error_dir / raw_file.name
            category = "raw_error"
        elif result["usable_nav"]:
            destination = raw_w_nav_dir / raw_file.name
            category = "raw_w_nav"
        else:
            destination = raw_no_nav_dir / raw_file.name
            category = "raw_no_nav"

        transfer_path(raw_file, destination)

        print(
            f"{relpath(raw_file)} -> {category} | "
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

    raw_w_nav_dir = base_dir / "raw_w_nav"
    raw_no_nav_dir = base_dir / "raw_no_nav"
    raw_error_dir = base_dir / "raw_error"
    calibration_dir = base_dir / "calibration"
    other_dir = base_dir / "other"

    if READ_ONLY_MODE:
        print(
            "READ ONLY MODE: No directories will be created and no files "
            "or folders will be moved or copied."
        )
    else:
        raw_w_nav_dir.mkdir(exist_ok=True)
        raw_no_nav_dir.mkdir(exist_ok=True)
        raw_error_dir.mkdir(exist_ok=True)
        calibration_dir.mkdir(exist_ok=True)
        other_dir.mkdir(exist_ok=True)

    organize_non_raw_and_calibration_dirs(base_dir)
    df = process_raw_files(base_dir)

    summary_path = base_dir / "raw_nav_summary.csv"

    print("\nSummary counts:")
    print(f"Source directory: {relpath(base_dir)}")
    print(f"Total raw files processed: {len(df)}")
    print(f"Usable nav: {(df['usable_nav'] == True).sum() if not df.empty else 0}")
    print(
        f"No nav: "
        f"{((df['usable_nav'] == False) & (df['error'] == '')).sum() if not df.empty else 0}"
    )
    print(f"Errors: {(df['error'] != '').sum() if not df.empty else 0}")

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