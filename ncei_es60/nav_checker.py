"""
nav_checker.py

Author:
    Ashton Flinders
    andrealphus@gmail.com

Date Created:
    2026-04-09

Date Modified:
    2026-06-24

Description:
    Module for organizing and classifying Simrad ES60 *.raw echosounder files
    based on the presence of usable navigation data. This module is designed to
    be imported by scripts or notebooks in the ncei_es60 package.

    The module performs the following tasks within a user-specified source
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

        6. Generates a summary table containing diagnostic information for
           each file. By default, the summary file is named raw_nav_summary.csv,
           but this can be changed with NavCheckerConfig.summary_filename.

Notes:
    - Although this module specifies sonar_model="EK60" when opening files
      with echopype, it is intended for processing Simrad ES60 data. The ES60
      and EK60 echosounders share the same .raw file format, and echopype uses
      the "EK60" designation to support both systems.

Typical notebook usage:

    from pathlib import Path
    from ncei_es60.nav_checker import NavCheckerConfig, process_cruise

    config = NavCheckerConfig(
        source_dir=Path("../CRUISES_raw/EBS17VA_example"),
        read_only_mode=False,
        move_files=False,
        output_dir=Path("../CRUISES_processed/EBS17VA_example"),
        use_sentence_for_nav=True,
    )

    summary_df = process_cruise(config)

Dependencies:
    - Python 3.10+
    - echopype
    - numpy
    - pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import sys

import echopype as ep
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================


@dataclass(slots=True)
class NavCheckerConfig:
    """User-editable configuration for ES60 raw-file navigation checking.

    Parameters
    ----------
    source_dir:
        Directory containing the original cruise files to organize and check.
        Only top-level *.raw files are processed by the navigation checker.

    read_only_mode:
        If True, inspect files and print what would happen, but do not create
        category directories and do not move or copy files. A summary CSV may
        still be written if write_summary_in_read_only is True.

    move_files:
        If read_only_mode is False, controls whether files are moved or copied.
        True moves files into category directories. False copies files to
        output_dir and leaves originals in place.

    output_dir:
        Destination base directory used when move_files is False. In read-only
        mode this is optional and is only used to show hypothetical output
        locations.

    create_output_dir:
        If True, create output_dir automatically when needed.

    write_summary_in_read_only:
        If True, write a diagnostic summary CSV even in read-only mode. This is
        the only allowed filesystem write during read-only operation.

    read_only_summary_dir:
        Optional destination directory for the summary CSV when in read-only
        mode. If None, the summary is written in source_dir.

    create_read_only_summary_dir:
        If True, create read_only_summary_dir automatically when needed.

    summary_filename:
        Optional custom filename for the summary CSV. If None, use
        raw_nav_summary.csv. If the supplied string lacks .csv, .csv is
        appended. The string may include {source_dir}, which is replaced with
        the source directory name.

    use_sentence_for_nav:
        If True, valid NMEA sentence_type content can be used as part of the
        decision that a file has usable navigation. If False, only valid
        latitude and longitude are used.
    """

    source_dir: Path | str
    read_only_mode: bool = True
    move_files: bool = False
    output_dir: Path | str | None = None
    create_output_dir: bool = True
    write_summary_in_read_only: bool = True
    read_only_summary_dir: Path | str | None = None
    create_read_only_summary_dir: bool = True
    summary_filename: str | None = None
    use_sentence_for_nav: bool = True

    @property
    def source_path(self) -> Path:
        """Return source_dir as an expanded Path."""
        return Path(self.source_dir).expanduser().resolve()

    @property
    def output_path(self) -> Path | None:
        """Return output_dir as an expanded Path, or None."""
        if self.output_dir is None:
            return None
        return Path(self.output_dir).expanduser().resolve()

    @property
    def read_only_summary_path(self) -> Path | None:
        """Return read_only_summary_dir as an expanded Path, or None."""
        if self.read_only_summary_dir is None:
            return None
        return Path(self.read_only_summary_dir).expanduser().resolve()


# =============================================================================
# General helpers
# =============================================================================


def get_summary_filename(base_dir: Path, config: NavCheckerConfig) -> str:
    """Return the resolved summary CSV filename."""
    if config.summary_filename is None:
        filename = "raw_nav_summary.csv"
    else:
        filename = str(config.summary_filename).format(source_dir=base_dir.name)

    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    return filename


def relpath(path: Path) -> str:
    """Return a path relative to the current working directory when possible."""
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


# =============================================================================
# Raw-file classification
# =============================================================================


def classify_raw_file(raw_file: Path, config: NavCheckerConfig) -> dict[str, object]:
    """Open a raw file and classify whether it contains usable nav data."""
    result: dict[str, object] = {
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
        "use_sentence_for_nav": config.use_sentence_for_nav,
        "error": "",
    }

    try:
        ed = ep.open_raw(str(raw_file), sonar_model="EK60")
        platform = ed["Platform"]

        lat = platform["latitude"].values if "latitude" in platform.data_vars else None
        lon = platform["longitude"].values if "longitude" in platform.data_vars else None
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

        if config.use_sentence_for_nav:
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

    except Exception as exc:  # noqa: BLE001 - classification should keep going.
        result["error"] = str(exc)

    return result


# =============================================================================
# Filesystem operation helpers
# =============================================================================


def transfer_path(source: Path, destination: Path, config: NavCheckerConfig) -> None:
    """Copy or move a file or directory depending on configuration."""
    if config.read_only_mode:
        print(f"READ ONLY: {relpath(source)} -> {relpath(destination)}")
        return

    if destination.exists():
        raise FileExistsError(f"Destination already exists: {relpath(destination)}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if config.move_files:
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


def prepare_destination_base_dir(
    base_dir: Path,
    config: NavCheckerConfig,
) -> Path:
    """Determine and validate the destination base directory."""
    if config.read_only_mode:
        if config.output_path is None:
            return base_dir

        # In read-only mode, output_dir is optional and only used to show where
        # files would go. It is not created here.
        return config.output_path

    if not config.move_files and config.output_path is None:
        raise ValueError(
            "output_dir must be specified when "
            "read_only_mode=False and move_files=False."
        )

    if config.move_files and config.output_path is not None:
        raise ValueError("output_dir should be None when move_files=True.")

    if config.move_files:
        return base_dir

    destination_base_dir = config.output_path
    assert destination_base_dir is not None

    if destination_base_dir.exists():
        if not destination_base_dir.is_dir():
            raise NotADirectoryError(
                f"Output path exists but is not a directory: "
                f"{relpath(destination_base_dir)}"
            )
        return destination_base_dir

    if config.create_output_dir:
        destination_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {relpath(destination_base_dir)}")
        return destination_base_dir

    raise FileNotFoundError(
        f"Output directory does not exist: {relpath(destination_base_dir)}"
    )


def prepare_summary_path(
    base_dir: Path,
    destination_base_dir: Path,
    config: NavCheckerConfig,
) -> Path | None:
    """Return summary CSV path, or None if no summary should be written."""
    if config.read_only_mode:
        if not config.write_summary_in_read_only:
            return None

        summary_dir = (
            base_dir
            if config.read_only_summary_path is None
            else config.read_only_summary_path
        )

        if summary_dir.exists():
            if not summary_dir.is_dir():
                raise NotADirectoryError(
                    f"Read-only summary path exists but is not a directory: "
                    f"{relpath(summary_dir)}"
                )
        elif config.create_read_only_summary_dir:
            summary_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created read-only summary directory: {relpath(summary_dir)}")
        else:
            raise FileNotFoundError(
                f"Read-only summary directory does not exist: {relpath(summary_dir)}"
            )

        return summary_dir / get_summary_filename(base_dir, config)

    return destination_base_dir / get_summary_filename(base_dir, config)


def create_category_dirs(
    destination_base_dir: Path,
    config: NavCheckerConfig,
) -> None:
    """Create output category directories if filesystem changes are enabled."""
    category_dirs = [
        destination_base_dir / "raw_w_nav",
        destination_base_dir / "raw_no_nav",
        destination_base_dir / "raw_error",
        destination_base_dir / "calibration",
        destination_base_dir / "other",
    ]

    if config.read_only_mode:
        for category_dir in category_dirs:
            print(f"READ ONLY: Would create directory: {relpath(category_dir)}")
        return

    for category_dir in category_dirs:
        category_dir.mkdir(parents=True, exist_ok=True)


def organize_non_raw_and_calibration_dirs(
    base_dir: Path,
    destination_base_dir: Path,
    config: NavCheckerConfig,
) -> None:
    """Organize top-level non-.raw files and calibration-like directories."""
    other_dir = destination_base_dir / "other"
    calibration_dir = destination_base_dir / "calibration"

    protected_names = {
        "raw_w_nav",
        "raw_no_nav",
        "raw_error",
        "calibration",
        "other",
        "raw_nav_summary.csv",
        get_summary_filename(base_dir, config),
    }

    for item in sorted(base_dir.iterdir()):
        if item.name in protected_names:
            continue

        if item.resolve() == destination_base_dir.resolve():
            continue

        if item.is_file() and item.suffix.lower() != ".raw":
            destination = other_dir / item.name
            transfer_path(item, destination, config)
            print(f"{relpath(item)} -> {relpath(destination)}")

        elif item.is_dir() and is_calibration_dir(item):
            destination = calibration_dir / item.name
            transfer_path(item, destination, config)
            print(f"{relpath(item)} -> {relpath(destination)}")


def process_raw_files(
    base_dir: Path,
    destination_base_dir: Path,
    config: NavCheckerConfig,
) -> pd.DataFrame:
    """Classify remaining top-level raw files and organize them by nav status."""
    raw_w_nav_dir = destination_base_dir / "raw_w_nav"
    raw_no_nav_dir = destination_base_dir / "raw_no_nav"
    raw_error_dir = destination_base_dir / "raw_error"

    raw_files = sorted(
        [
            path
            for path in base_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".raw"
        ]
    )

    results: list[dict[str, object]] = []

    for raw_file in raw_files:
        result = classify_raw_file(raw_file, config)

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

        transfer_path(raw_file, destination, config)

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


# =============================================================================
# Public API
# =============================================================================


def process_cruise(config: NavCheckerConfig) -> pd.DataFrame:
    """Organize source directory contents and classify raw files by nav status."""
    base_dir = config.source_path

    if not base_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {relpath(base_dir)}")

    if not base_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {relpath(base_dir)}")

    destination_base_dir = prepare_destination_base_dir(base_dir, config)

    if config.read_only_mode:
        print(
            "READ ONLY MODE: No category directories will be created and "
            "no files or folders will be moved or copied."
        )

    print(f"Source directory: {relpath(base_dir)}")
    print(f"Destination directory: {relpath(destination_base_dir)}")
    print(f"read_only_mode: {config.read_only_mode}")
    print(f"move_files: {config.move_files}")
    print(f"create_output_dir: {config.create_output_dir}")
    print(f"write_summary_in_read_only: {config.write_summary_in_read_only}")
    print(f"read_only_summary_dir: {config.read_only_summary_dir}")
    print(f"summary_filename: {config.summary_filename}")
    print(f"Resolved summary filename: {get_summary_filename(base_dir, config)}")

    create_category_dirs(destination_base_dir, config)

    organize_non_raw_and_calibration_dirs(
        base_dir=base_dir,
        destination_base_dir=destination_base_dir,
        config=config,
    )

    df = process_raw_files(
        base_dir=base_dir,
        destination_base_dir=destination_base_dir,
        config=config,
    )

    summary_path = prepare_summary_path(
        base_dir=base_dir,
        destination_base_dir=destination_base_dir,
        config=config,
    )

    print("\nSummary counts:")
    print(f"Source directory: {relpath(base_dir)}")
    print(f"Destination directory: {relpath(destination_base_dir)}")
    print(f"Total raw files processed: {len(df)}")
    print(f"Usable nav: {(df['usable_nav'] == True).sum() if not df.empty else 0}")
    print(
        "No nav: "
        f"{((df['usable_nav'] == False) & (df['error'] == '')).sum() if not df.empty else 0}"
    )
    print(f"Errors: {(df['error'] != '').sum() if not df.empty else 0}")

    if summary_path is None:
        print("READ ONLY: Summary CSV not written.")
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)
        print(f"Summary written to: {relpath(summary_path)}")

    return df


# Alias matching the depth_extractor.py pattern.
def main(config: NavCheckerConfig) -> pd.DataFrame:
    """Run navigation checking using a NavCheckerConfig."""
    return process_cruise(config)


# =============================================================================
# Optional command-line entry point
# =============================================================================


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

    cli_config = NavCheckerConfig(source_dir=source_dir)

    try:
        summary_df = process_cruise(cli_config)
    except Exception as exc:  # noqa: BLE001 - command-line output should be concise.
        print(f"Error: {exc}")
        sys.exit(1)

    print("\nSummary table:")
    print(summary_df)
