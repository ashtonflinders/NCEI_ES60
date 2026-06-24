"""
ES60 / EK60 bottom-depth extraction from Simrad .raw files.

Purpose
-------
This standalone script processes one ES60/EK60 .raw file, or all .raw files in a
single directory, and exports bottom-depth detections for each available channel
and each enabled bottom-detection method.

The script is intended as a clear, editable processing prototype for extracting
water-column-derived bottom depths from fisheries echosounder data using
Echopype.

Major processing steps
----------------------
1. Read one or more EK60 .raw files with Echopype.
2. Compute Sv.
3. Add transducer-relative depth.
4. Add navigation/location from parsed NMEA records.
5. Run the Echopype basic seafloor detector for each channel.
6. If split-beam angle variables are present, run the Echopype Blackwell
   detector for each channel.
7. Export one combined CSV file per input file, with one depth column per frequency/method.
8. Optionally save a QC echogram image showing detected bottom picks.
9. Append failed input files to an error log in the top-level output directory.

Input behavior
--------------
CRUISE_DIR is the top-level cruise processing directory, for example:
    ../CRUISES_processed/EBS17VA

The script expects navigation-processed raw files under:
    CRUISE_DIR / "raw_w_nav"

Set SINGLE_RAW_FILE to a filename or Path to process one file. Set it to None
to process all files matching RAW_GLOB in RAW_DIR.

Output behavior
---------------
All CSV outputs are written to:
    OUTPUT_DIR

If SAVE_DEPTH_IMAGES is True, QC images are written to:
    CRUISE_DIR / "images"

Any file that fails is appended to:
    CRUISE_DIR / ERROR_LOG_NAME

CSV naming convention
---------------------
For an input file such as:
    L0004-D20170602-T224458-ES60.raw

the combined CSV output will be named:
    L0004-D20170602-T224458-ES60_depth.csv

CSV columns
-----------
Each output CSV contains shared navigation columns plus one sound-speed column
per frequency and one bottom-depth column per frequency/method, for example:
    * ping_time
    * latitude
    * longitude
    * 38kHz_sound_speed_m_s
    * 38kHz_basic
    * 38kHz_blackwell
    * 120kHz_sound_speed_m_s
    * 120kHz_basic
    * 120kHz_blackwell

Note: 120000 Hz is written as 120kHz. If a frequency is not an exact kHz value,
the column tag falls back to a sanitized Hz-based label.

Dependencies
------------
Required:
    echopype
    numpy
    pandas
    xarray
    matplotlib

Optional:
    hvplot.xarray is imported only to preserve compatibility with the original
    interactive workflow. It is not required for batch CSV/image export.

Adapted from
------------
Echopype seafloor detection example:
https://echopype-examples.readthedocs.io/en/latest/seafloor_detection.html
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import traceback

import echopype as ep
from echopype.mask import detect_seafloor
import hvplot.xarray  # noqa: F401  # Enables interactive xarray/hvplot support.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# User settings
# =============================================================================

# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------

# Cruise-level directory containing:
#
#   CRUISE_DIR/
#     raw_w_nav/
#       *.raw
#     output/
#       *.csv
#     images/
#       *.png
#     depth_processing_errors.txt
#
# Example:
#   ../CRUISES_processed/EBS17VA
#
CRUISE_DIR = Path(
    "../CRUISES_processed/EBS17VA_example"
)

# Directory containing navigation-processed ES60 raw files.
RAW_DIR = CRUISE_DIR / "raw_w_nav"

# Optional: process a single file instead of all files in RAW_DIR.
# Set to None to process all matching files.
SINGLE_RAW_FILE = "L0004-D20170603-T001306-ES60.raw"

# Example:
# SINGLE_RAW_FILE = (
#     RAW_DIR /
#     "L0004-D20170602-T224458-ES60.raw"
# )

# File pattern used when processing an entire directory.
RAW_GLOB = "*.raw"

# Output locations.
OUTPUT_DIR = CRUISE_DIR / "depth"
IMAGE_DIR = CRUISE_DIR / "images"

# Error log written at the cruise level.
ERROR_LOG = CRUISE_DIR / "depth_processing_errors.txt"

# Export one CSV per input file, per channel, per detection method.
EXPORT_BOTTOM_DEPTH_CSV = False

# Save QC echogram images with detected bottom-depth lines.
# Images are saved only if this is True.
SAVE_DEPTH_IMAGES = False

# Bottom-detection methods.
# Basic thresholding is always the default method and always runs.
# Set RUN_BLACKWELL_DETECTOR to True to additionally run the Blackwell detector
# when split-beam angle data are available in the input file.
RUN_BLACKWELL_DETECTOR = False

# Image format for saved depth QC plots. Common options: "png", "pdf", "jpg".
DEPTH_IMAGE_FORMAT = "png"
DEPTH_IMAGE_DPI = 200

# Show figures interactively while the script runs. For batch processing, this
# should usually remain False.
SHOW_PLOTS = False

# Optional diagnostic plots from the original exploratory script. These are
# useful during development, but should usually remain False for batch runs.
PLOT_SV = False
PLOT_BASIC_SEAFLOOR = False
PLOT_BLACKWELL_DEPTH = False
PLOT_BLACKWELL_ANGLES = False
PLOT_COMPARE_BOTTOM_DETECTIONS = False

# Number of near-surface range bins to skip for the basic detector. This helps
# avoid false detections in the saturated surface zone.
N_SKIP = 20


# =============================================================================
# Basic bottom-detection settings
# =============================================================================

BASIC_THRESHOLD = (-40, -20)
BASIC_OFFSET_M = 0.3


# =============================================================================
# Blackwell bottom-detection settings
# =============================================================================

BLACKWELL_THRESHOLD = [-40, 702, 282]
BLACKWELL_OFFSET = 0.3
BLACKWELL_R0 = 10
BLACKWELL_R1 = 1000
BLACKWELL_WTHETA = 28
BLACKWELL_WPHI = 52


# =============================================================================
# Path helpers
# =============================================================================


def get_output_dirs() -> tuple[Path, Path, Path]:
    """Return output, image, and error-log paths and create needed directories."""
    output_dir = OUTPUT_DIR
    image_dir = IMAGE_DIR
    error_log = ERROR_LOG

    output_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_DEPTH_IMAGES:
        image_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, image_dir, error_log



def discover_raw_files() -> list[Path]:
    """
    Return a list of raw files to process.

    If SINGLE_RAW_FILE is specified, only that file is processed. SINGLE_RAW_FILE
    may be either an absolute/relative Path or just a filename located in RAW_DIR.
    Otherwise all files matching RAW_GLOB in RAW_DIR are processed.
    """

    if SINGLE_RAW_FILE is not None:
        single_raw_file = Path(SINGLE_RAW_FILE)

        # If only a filename was supplied, interpret it relative to RAW_DIR.
        if not single_raw_file.is_absolute() and single_raw_file.parent == Path("."):
            single_raw_file = RAW_DIR / single_raw_file

        return [single_raw_file]

    return sorted(RAW_DIR.glob(RAW_GLOB))



def append_error(error_log: Path, raw_file: Path, error: BaseException) -> None:
    """Append a failed filename and traceback to the top-level error log."""
    timestamp = datetime.now().isoformat(timespec="seconds")

    with error_log.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Failed file: {raw_file}\n")
        f.write(f"Error type: {type(error).__name__}\n")
        f.write(f"Error message: {error}\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        f.write("\n")


# =============================================================================
# Naming helpers
# =============================================================================


def sanitize_for_filename(value: object) -> str:
    """Convert a value into a compact filename-safe string."""
    text = str(value)
    text = text.strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("_")
    return text or "unknown"



def frequency_to_short_label(frequency_hz: float) -> str:
    """Return a short frequency label such as 38kHz or 120kHz."""
    if not np.isfinite(frequency_hz):
        return "unknownHz"

    frequency_hz_int = int(round(float(frequency_hz)))

    # Most ES60/EK60 nominal frequencies are clean kHz values.
    if frequency_hz_int % 1000 == 0:
        return f"{frequency_hz_int // 1000}kHz"

    return f"{frequency_hz_int}Hz"



def output_csv_path(
    raw_file: Path,
    output_dir: Path,
    frequency_hz: float,
    method: str,
) -> Path:
    """Build the per-file/per-channel/per-method output CSV path."""
    freq_label = frequency_to_short_label(frequency_hz)
    method_label = sanitize_for_filename(method.lower())
    return output_dir / f"{raw_file.stem}_{freq_label}_{method_label}.csv"



def output_image_path(
    raw_file: Path,
    image_dir: Path,
    frequency_hz: float,
) -> Path:
    """Build the per-file/per-channel QC image path."""
    freq_label = frequency_to_short_label(frequency_hz)
    ext = DEPTH_IMAGE_FORMAT.lower().lstrip(".")
    return image_dir / f"{raw_file.stem}_{freq_label}_bottom_detection.{ext}"


# =============================================================================
# Data preparation helpers
# =============================================================================

def get_sound_speed_for_channel(
    ed: object,
    channel: object,
) -> xr.DataArray | None:
    """Return channel-specific sound speed aligned to ping_time, if available.

    For EK60/ES60 files read by Echopype, the sound speed is commonly stored in
    ed["Environment"] as sound_speed_indicative with dimensions such as
    (channel, time1). The bottom detections and navigation use ping_time, so this
    helper selects the requested channel and renames the Environment time
    dimension to ping_time before alignment/export.
    """

    possible_names = [
        "sound_speed_indicative",
        "sound_speed",
        "sound_velocity",
    ]

    try:
        env = ed["Environment"]
    except Exception:
        return None

    for name in possible_names:
        if name not in env:
            continue

        ss = env[name]

        # Select the same channel being exported.
        if "channel" in ss.dims:
            ss = ss.sel(channel=channel)

        # Rename the Environment time dimension to ping_time so xr.align() can
        # align sound speed with the bottom-depth picks and navigation.
        time_dims = [dim for dim in ss.dims if dim == "ping_time"]
        if not time_dims:
            time_dims = [dim for dim in ss.dims if dim.lower().startswith("time")]

        if time_dims and time_dims[0] != "ping_time":
            ss = ss.rename({time_dims[0]: "ping_time"})

        return ss

    return None

    
def add_splitbeam_angles_if_available(
    ds_sv: xr.Dataset,
    beam_group: xr.Dataset,
) -> tuple[xr.Dataset, bool, xr.DataArray | None, xr.DataArray | None]:
    """Attach split-beam angle variables when they are present in the raw file."""
    required_vars = [
        "angle_alongship",
        "angle_athwartship",
    ]

    missing_vars = [var for var in required_vars if var not in beam_group]

    if missing_vars:
        print("Missing required split-beam variables:")
        for var in missing_vars:
            print(f"  {var}")
        return ds_sv, False, None, None

    print("Split-beam angle variables found.")

    angle_along = beam_group["angle_alongship"]
    angle_athwart = beam_group["angle_athwartship"]

    ds_sv = ds_sv.assign(
        angle_alongship=angle_along,
        angle_athwartship=angle_athwart,
    )

    return ds_sv, True, angle_along, angle_athwart



def check_navigation(ds_sv: xr.Dataset) -> bool:
    """Return True if latitude and longitude are available in ds_sv."""
    required_nav_vars = [
        "latitude",
        "longitude",
    ]

    missing_nav_vars = [var for var in required_nav_vars if var not in ds_sv]

    if missing_nav_vars:
        print("Missing required navigation variables:")
        for var in missing_nav_vars:
            print(f"  {var}")
        return False

    print("Navigation variables found.")
    print("latitude dims:", ds_sv["latitude"].dims)
    print("longitude dims:", ds_sv["longitude"].dims)
    return True


# =============================================================================
# Bottom detection helpers
# =============================================================================


def run_basic_detector(ds_sv: xr.Dataset, channel: object) -> xr.DataArray:
    """Run Echopype's basic seafloor detector for one channel."""
    depth = detect_seafloor(
        ds_sv,
        method="basic",
        params={
            "var_name": "Sv",
            "channel": channel,
            "threshold": BASIC_THRESHOLD,
            "offset_m": BASIC_OFFSET_M,
            "bin_skip_from_surface": N_SKIP,
        },
    )

    if not isinstance(depth, xr.DataArray):
        raise TypeError("Basic detector did not return an xarray.DataArray.")

    if set(depth.dims) != {"ping_time"}:
        raise ValueError(f"Unexpected basic depth dimensions: {depth.dims}")

    return depth



def run_blackwell_detector(ds_sv: xr.Dataset, channel: object) -> xr.DataArray:
    """Run Echopype's Blackwell seafloor detector for one channel."""
    depth = detect_seafloor(
        ds=ds_sv,
        method="blackwell",
        params={
            "channel": channel,
            "var_name": "Sv",
            "threshold": BLACKWELL_THRESHOLD,
            "offset": BLACKWELL_OFFSET,
            "r0": BLACKWELL_R0,
            "r1": BLACKWELL_R1,
            "wtheta": BLACKWELL_WTHETA,
            "wphi": BLACKWELL_WPHI,
        },
    )

    if not isinstance(depth, xr.DataArray):
        raise TypeError("Blackwell detector did not return an xarray.DataArray.")

    if set(depth.dims) != {"ping_time"}:
        raise ValueError(f"Unexpected Blackwell depth dimensions: {depth.dims}")

    return depth



def run_bottom_detection(
    ds_sv: xr.Dataset,
    channels_available: np.ndarray,
    has_splitbeam_angles: bool,
) -> dict[object, dict[str, xr.DataArray]]:
    """Run all enabled bottom-detection methods for each available channel."""
    detections: dict[object, dict[str, xr.DataArray]] = {}

    for channel in channels_available:
        print(f"\nRunning bottom detection for: {channel}")

        detections[channel] = {}
        detections[channel]["basic"] = run_basic_detector(ds_sv, channel)

        if RUN_BLACKWELL_DETECTOR and has_splitbeam_angles:
            detections[channel]["blackwell"] = run_blackwell_detector(ds_sv, channel)
        elif RUN_BLACKWELL_DETECTOR and not has_splitbeam_angles:
            print("  Skipping Blackwell detector because split-beam angles are missing.")

    return detections


# =============================================================================
# CSV export helpers
# =============================================================================


def output_combined_csv_path(
    raw_file: Path,
    output_dir: Path,
) -> Path:
    """Build the one-file-per-raw-file combined depth CSV path."""
    return output_dir / f"{raw_file.stem}_depth.csv"



def make_navigation_dataframe(
    lat: xr.DataArray,
    lon: xr.DataArray,
) -> pd.DataFrame:
    """Build the base navigation table used by the combined CSV."""
    lat_aligned, lon_aligned = xr.align(
        lat,
        lon,
        join="inner",
    )

    if not lat_aligned.ping_time.equals(lon_aligned.ping_time):
        raise ValueError("Latitude and longitude ping times are not aligned.")

    return pd.DataFrame(
        {
            "ping_time": lat_aligned["ping_time"].values,
            "latitude": np.round(lat_aligned.values, 6),
            "longitude": np.round(lon_aligned.values, 6),
        }
    )



def make_depth_column_dataframe(
    depth: xr.DataArray,
    column_name: str,
) -> pd.DataFrame:
    """Build a two-column DataFrame for one bottom-depth detection series."""
    return pd.DataFrame(
        {
            "ping_time": depth["ping_time"].values,
            column_name: np.round(depth.values, 2),
        }
    )



def make_sound_speed_column_dataframe(
    sound_speed: xr.DataArray | None,
    column_name: str,
) -> pd.DataFrame | None:
    """Build a two-column DataFrame for one channel's sound-speed series."""
    if sound_speed is None:
        return None

    ss_values = np.asarray(sound_speed.values)

    # Scalar sound speed has no ping_time coordinate. In that case, the value is
    # handled later by broadcasting onto the combined table.
    if ss_values.ndim == 0:
        return pd.DataFrame(
            {
                "ping_time": [],
                column_name: [],
            }
        ).assign(_scalar_sound_speed=float(ss_values))

    if "ping_time" not in sound_speed.dims:
        print(
            f"Cannot export {column_name}; sound speed has no ping_time dimension."
        )
        return None

    return pd.DataFrame(
        {
            "ping_time": sound_speed["ping_time"].values,
            column_name: np.round(ss_values, 2),
        }
    )



def merge_on_ping_time(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-merge two tables on ping_time without dropping unmatched pings."""
    return left.merge(
        right,
        on="ping_time",
        how="outer",
    )



def export_combined_bottom_depth_csv(
    raw_file: Path,
    output_dir: Path,
    ed: object,
    ds_sv: xr.Dataset,
    detections: dict[object, dict[str, xr.DataArray]],
) -> None:
    """Export one combined CSV per raw file.

    The combined CSV keeps navigation in shared columns and writes each detected
    depth series as its own column, using frequency and method in the column
    name. For example, a two-frequency file with basic and Blackwell enabled
    will produce columns such as:

        ping_time
        latitude
        longitude
        38kHz_sound_speed_m_s
        38kHz_basic
        38kHz_blackwell
        120kHz_sound_speed_m_s
        120kHz_basic
        120kHz_blackwell

    An outer merge is used for detection columns so that pings are not silently
    dropped if one method or frequency returns fewer bottom picks than another.
    """
    if not EXPORT_BOTTOM_DEPTH_CSV:
        return

    if not check_navigation(ds_sv):
        print("Cannot export bottom-depth CSV because navigation is missing.")
        return

    lat = ds_sv["latitude"]
    lon = ds_sv["longitude"]
    frequency_nominal = ds_sv["frequency_nominal"]

    combined_df = make_navigation_dataframe(lat, lon)

    for channel, method_depths in detections.items():
        frequency_hz = float(frequency_nominal.sel(channel=channel).values)
        freq_label = frequency_to_short_label(frequency_hz)

        sound_speed_channel = get_sound_speed_for_channel(
            ed=ed,
            channel=channel,
        )

        sound_speed_col = f"{freq_label}_sound_speed_m_s"
        ss_df = make_sound_speed_column_dataframe(
            sound_speed=sound_speed_channel,
            column_name=sound_speed_col,
        )

        if ss_df is not None:
            if "_scalar_sound_speed" in ss_df.columns:
                combined_df[sound_speed_col] = round(
                    float(ss_df["_scalar_sound_speed"].iloc[0]),
                    2,
                )
            else:
                combined_df = merge_on_ping_time(combined_df, ss_df)

        for method, depth in method_depths.items():
            depth_col = f"{freq_label}_{sanitize_for_filename(method.lower())}"
            depth_df = make_depth_column_dataframe(
                depth=depth,
                column_name=depth_col,
            )
            combined_df = merge_on_ping_time(combined_df, depth_df)

    combined_df = combined_df.sort_values("ping_time")

    # Re-round after merges so the exported CSV has the intended precision.
    if "latitude" in combined_df:
        combined_df["latitude"] = combined_df["latitude"].round(6)
    if "longitude" in combined_df:
        combined_df["longitude"] = combined_df["longitude"].round(6)

    for col in combined_df.columns:
        if col.endswith("_sound_speed_m_s") or col.endswith("_basic") or col.endswith("_blackwell"):
            combined_df[col] = combined_df[col].round(2)

    csv_path = output_combined_csv_path(
        raw_file=raw_file,
        output_dir=output_dir,
    )

    combined_df.to_csv(
        csv_path,
        index=False,
    )

    print(f"Saved combined bottom-depth CSV to: {csv_path}")


# =============================================================================
# Plotting and image helpers
# =============================================================================


def maybe_show_or_close(fig: plt.Figure) -> None:
    """Show a figure interactively or close it to avoid memory buildup."""
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)



def plot_sv_diagnostics(ds_sv: xr.Dataset, n_channels: int) -> None:
    """Optional exploratory Sv plots from the original script."""
    if not PLOT_SV:
        return

    # Full Sv range.
    plot_obj = ds_sv["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-80,
        vmax=-30,
        cmap="RdYlBu_r",
        yincrease=False,
    )
    maybe_show_or_close(plot_obj.fig)

    # Narrower Sv range to emphasize the seafloor return.
    plot_obj = ds_sv["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-40,
        vmax=-20,
        cmap="RdYlBu_r",
        yincrease=False,
    )
    maybe_show_or_close(plot_obj.fig)

    # First range samples only, to inspect the surface saturation zone.
    plot_obj = ds_sv.isel(range_sample=slice(0, 50))["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-40,
        vmax=-20,
        cmap="RdYlBu_r",
        yincrease=False,
        ylim=(50, 0),
    )
    maybe_show_or_close(plot_obj.fig)



def plot_simple_depth_diagnostics(
    detections: dict[object, dict[str, xr.DataArray]],
) -> None:
    """Optional simple line/point plots for bottom detections."""
    for channel, method_depths in detections.items():
        if PLOT_BASIC_SEAFLOOR and "basic" in method_depths:
            basic_depth = method_depths["basic"]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(
                basic_depth["ping_time"].values,
                basic_depth.values,
                ".",
                markersize=1,
            )
            ax.set_title(f"Basic seafloor depth: {channel}")
            ax.set_xlabel("Ping time")
            ax.set_ylabel("Depth (m)")
            ax.invert_yaxis()
            maybe_show_or_close(fig)

        if PLOT_BLACKWELL_DEPTH and "blackwell" in method_depths:
            blackwell_depth = method_depths["blackwell"]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(
                blackwell_depth["ping_time"].values,
                blackwell_depth.values,
                ".",
                label="Blackwell",
                color="orange",
                markersize=1,
            )
            ax.invert_yaxis()
            ax.set_title(f"Blackwell seafloor depth: {channel}")
            ax.set_xlabel("Ping time")
            ax.set_ylabel("Depth (m)")
            ax.legend(loc="upper right")
            ax.grid(True)
            maybe_show_or_close(fig)



def plot_blackwell_angle_diagnostics(
    channels_available: np.ndarray,
    angle_along: xr.DataArray | None,
    angle_athwart: xr.DataArray | None,
) -> None:
    """Optional split-beam angle plots from the original script."""
    if not PLOT_BLACKWELL_ANGLES:
        return

    if angle_along is None or angle_athwart is None:
        return

    for channel in channels_available:
        angle_athwart_sel = angle_athwart.sel(channel=channel)
        angle_along_sel = angle_along.sel(channel=channel)

        fig, axes = plt.subplots(ncols=2, figsize=(14, 5), sharey=True)

        angle_athwart_sel.plot(
            ax=axes[0],
            x="ping_time",
            y="range_sample",
            cmap="RdBu",
            yincrease=False,
            robust=True,
            cbar_kwargs={"label": "Athwart angle (deg)"},
        )
        axes[0].set_title("Athwartship angle")

        angle_along_sel.plot(
            ax=axes[1],
            x="ping_time",
            y="range_sample",
            cmap="RdBu",
            yincrease=False,
            robust=True,
            cbar_kwargs={"label": "Alongship angle (deg)"},
        )
        axes[1].set_title("Alongship angle")

        fig.suptitle(f"Channel: {channel}")
        plt.tight_layout()
        maybe_show_or_close(fig)



def plot_compare_bottom_detections(
    detections: dict[object, dict[str, xr.DataArray]],
) -> None:
    """Optional comparison plot for basic and Blackwell detections."""
    if not PLOT_COMPARE_BOTTOM_DETECTIONS:
        return

    for channel, method_depths in detections.items():
        if "basic" not in method_depths or "blackwell" not in method_depths:
            continue

        basic_depth = method_depths["basic"]
        blackwell_depth = method_depths["blackwell"]

        pt_basic = basic_depth.ping_time
        pt_blackwell = blackwell_depth.ping_time

        missing_in_blackwell = pt_basic[~pt_basic.isin(pt_blackwell)]
        missing_in_basic = pt_blackwell[~pt_blackwell.isin(pt_basic)]

        print(f"\nComparison for: {channel}")
        print(
            "Ping times in basic_depth but missing "
            f"in blackwell_depth: {missing_in_blackwell.size}"
        )
        print(missing_in_blackwell.values)
        print(
            "Ping times in blackwell_depth but missing "
            f"in basic_depth: {missing_in_basic.size}"
        )
        print(missing_in_basic.values)

        bd, bw = xr.align(basic_depth, blackwell_depth, join="inner")

        if not bd.ping_time.equals(bw.ping_time):
            raise ValueError("Basic and Blackwell ping times are not aligned.")

        diff = bd - bw
        common_time = bd.ping_time

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharex=True)

        axs[0].plot(common_time, bd, ".", label="Basic", color="navy", markersize=1)
        axs[0].plot(
            common_time,
            bw,
            ".",
            label="Blackwell",
            color="orange",
            markersize=1,
        )
        axs[0].invert_yaxis()
        axs[0].set_title(f"Seafloor depth: {channel}")
        axs[0].set_xlabel("Ping time")
        axs[0].set_ylabel("Depth (m)")
        axs[0].legend(loc="upper right")
        axs[0].grid(True)

        axs[1].plot(common_time, diff, ".", color="darkgreen", markersize=1)
        axs[1].axhline(0, color="gray", linestyle="--", linewidth=1)
        axs[1].set_title("Difference: Basic – Blackwell")
        axs[1].set_xlabel("Ping time")
        axs[1].set_ylabel("Depth difference (m)")
        axs[1].grid(True)

        plt.tight_layout()
        maybe_show_or_close(fig)



def make_echogram_bottom_figure(
    ds_sv: xr.Dataset,
    channel: object,
    method_depths: dict[str, xr.DataArray],
) -> plt.Figure:
    """Create a QC echogram figure with detected bottom lines for one channel."""
    sv_da = ds_sv["Sv"].sel(channel=channel)
    depth = ds_sv["depth"].sel(channel=channel).isel(ping_time=0)

    finite_depth = np.isfinite(depth.values)

    sv_da_finite = sv_da.isel(range_sample=finite_depth)
    depth_finite = depth.values[finite_depth]

    # Sort depth so pcolormesh plots correctly even if the native range/depth
    # coordinates are not monotonic.
    sort_idx = np.argsort(depth_finite)

    sv_values_sorted = sv_da_finite.values[:, sort_idx]
    depth_sorted = depth_finite[sort_idx]

    sv_plot = xr.DataArray(
        data=sv_values_sorted,
        dims=["ping_time", "depth"],
        coords={
            "ping_time": sv_da["ping_time"].values,
            "depth": depth_sorted,
        },
        name="Sv",
    ).expand_dims(channel=[channel])

    ds_single = xr.Dataset({"Sv": sv_plot})

    fig, ax = plt.subplots(figsize=(20, 6))

    ds_single["Sv"].isel(channel=0).plot.pcolormesh(
        ax=ax,
        x="ping_time",
        y="depth",
        cmap="RdYlBu_r",
        yincrease=False,
        vmin=-80,
        vmax=-40,
        alpha=0.6,
        infer_intervals=False,
        cbar_kwargs={"label": "Sv (dB)"},
    )

    if "basic" in method_depths:
        basic_depth = method_depths["basic"]
        ax.plot(
            basic_depth["ping_time"].values,
            basic_depth.values,
            color="cyan",
            label="Basic bottom",
            linewidth=1.0,
        )

    if "blackwell" in method_depths:
        blackwell_depth = method_depths["blackwell"]
        ax.plot(
            blackwell_depth["ping_time"].values,
            blackwell_depth.values,
            color="orange",
            label="Blackwell bottom",
            linewidth=1.5,
        )

    ax.set_title(f"Echogram and detected seafloor: {channel}")
    ax.set_xlabel("Ping time")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig



def save_depth_images(
    raw_file: Path,
    image_dir: Path,
    ds_sv: xr.Dataset,
    detections: dict[object, dict[str, xr.DataArray]],
) -> None:
    """Save one QC echogram image per input file and channel."""
    if not SAVE_DEPTH_IMAGES:
        return

    frequency_nominal = ds_sv["frequency_nominal"]

    for channel, method_depths in detections.items():
        frequency_hz = float(frequency_nominal.sel(channel=channel).values)
        fig = make_echogram_bottom_figure(ds_sv, channel, method_depths)
        image_path = output_image_path(raw_file, image_dir, frequency_hz)

        fig.savefig(
            image_path,
            dpi=DEPTH_IMAGE_DPI,
            bbox_inches="tight",
        )
        print(f"Saved depth QC image to: {image_path}")
        maybe_show_or_close(fig)


# =============================================================================
# Per-file processing
# =============================================================================


def process_raw_file(raw_file: Path, output_dir: Path, image_dir: Path) -> None:
    """Process one EK60 .raw file and write requested outputs."""
    print("\n" + "=" * 80)
    print(f"Processing raw file: {raw_file}")

    # -------------------------------------------------------------------------
    # Read raw EK60 file.
    # -------------------------------------------------------------------------
    ed = ep.open_raw(raw_file, sonar_model="EK60")

    beam_group = ed["Sonar/Beam_group1"]
    channels_available = beam_group["channel"].values
    n_channels = len(channels_available)

    print("Available channels:")
    for channel in channels_available:
        print(f"  {channel}")

    # -------------------------------------------------------------------------
    # Compute Sv and add depth/location.
    # -------------------------------------------------------------------------
    ds_sv = ep.calibrate.compute_Sv(ed)

    # Add depth coordinates relative to the transducer.
    ds_sv = ep.consolidate.add_depth(ds_sv, ed, depth_offset=0)

    # Add position information from the parsed NMEA/navigation records.
    ds_sv = ep.consolidate.add_location(ds_sv, ed)

    # -------------------------------------------------------------------------
    # Optional diagnostics and split-beam preparation.
    # -------------------------------------------------------------------------
    plot_sv_diagnostics(ds_sv, n_channels)

    ds_sv, has_splitbeam_angles, angle_along, angle_athwart = (
        add_splitbeam_angles_if_available(ds_sv, beam_group)
    )

    # -------------------------------------------------------------------------
    # Run bottom detection for each channel.
    # -------------------------------------------------------------------------
    detections = run_bottom_detection(
        ds_sv=ds_sv,
        channels_available=channels_available,
        has_splitbeam_angles=has_splitbeam_angles,
    )

    # -------------------------------------------------------------------------
    # Optional diagnostic plots.
    # -------------------------------------------------------------------------
    plot_simple_depth_diagnostics(detections)
    plot_blackwell_angle_diagnostics(channels_available, angle_along, angle_athwart)
    plot_compare_bottom_detections(detections)

    # -------------------------------------------------------------------------
    # Export requested products.
    # -------------------------------------------------------------------------
    export_combined_bottom_depth_csv(raw_file, output_dir, ed, ds_sv, detections)
    save_depth_images(raw_file, image_dir, ds_sv, detections)

    print(f"Finished processing: {raw_file.name}")


# =============================================================================
# Script entry point
# =============================================================================


def main() -> None:
    """Process all requested raw files and keep going if individual files fail."""
    output_dir, image_dir, error_log = get_output_dirs()

    raw_files = discover_raw_files()

    if not raw_files:
        raise FileNotFoundError(f"No raw files found in RAW_DIR: {RAW_DIR}")

    print(f"Found {len(raw_files)} raw file(s) to process.")
    print(f"CSV output directory: {output_dir}")

    if SAVE_DEPTH_IMAGES:
        print(f"Image output directory: {image_dir}")

    print(f"Error log: {error_log}")

    n_success = 0
    n_failed = 0

    for raw_file in raw_files:
        try:
            process_raw_file(raw_file, output_dir, image_dir)
            n_success += 1
        except Exception as exc:  # noqa: BLE001 - batch script should continue.
            n_failed += 1
            print(f"ERROR: Failed processing {raw_file}")
            print(f"  {type(exc).__name__}: {exc}")
            append_error(error_log, raw_file, exc)

    print("\n" + "=" * 80)
    print("Processing complete.")
    print(f"Successful files: {n_success}")
    print(f"Failed files:     {n_failed}")

    if n_failed:
        print(f"See error log: {error_log}")


if __name__ == "__main__":
    main()
