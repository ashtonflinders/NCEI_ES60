"""
depth_extractor.py

Author:
    Ashton Flinders
    andrealphus@gmail.com

Date Created:
    2026-06-11

Description:
    Module for extracting bottom-depth picks from Simrad ES60/EK60 *.raw files
    that have already been sorted/processed for navigation. This module is
    designed to be imported by scripts or notebooks in the ncei_es60 package.

    The default expected cruise-directory structure is:

        CRUISE_DIR/
            raw_w_nav/
                *.raw
            depth/
                *_depth.csv
            images/
                *_bottom_detection.png
            depth_processing_errors.txt

    Each output CSV is one combined file per input *.raw file. The CSV contains
    shared navigation columns plus one sound-speed column per frequency and one
    bottom-depth column per frequency/method, for example:

        ping_time
        latitude
        longitude
        38kHz_sound_speed_m_s
        38kHz_basic
        38kHz_blackwell
        120kHz_sound_speed_m_s
        120kHz_basic
        120kHz_blackwell

Notes:
    - Although this module specifies sonar_model="EK60" when opening files with
      echopype, it is intended for processing Simrad ES60 data. ES60 and EK60
      share the same .raw file format, and echopype uses the "EK60" designation
      to support both systems.
    - Basic bottom detection always runs.
    - Blackwell detection is optional and requires split-beam angle variables.

Typical notebook usage:

    from pathlib import Path
    from ncei_es60.depth_extractor import DepthExtractionConfig, process_cruise

    config = DepthExtractionConfig(
        cruise_dir=Path("../CRUISES_processed/EBS17VA"),
        single_raw_file="L0004-D20170602-T224458-ES60.raw",
        run_blackwell_detector=True,
        save_depth_images=True,
    )

    summary_df = process_cruise(config)

Dependencies:
    - Python 3.10+
    - echopype
    - numpy
    - pandas
    - xarray
    - matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import re
import traceback

import echopype as ep
from echopype.mask import detect_seafloor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Configuration
# =============================================================================


@dataclass(slots=True)
class DepthExtractionConfig:
    """User-editable configuration for bottom-depth extraction.

    Parameters
    ----------
    cruise_dir:
        Top-level processed cruise directory. By default, raw input files are
        expected in cruise_dir / raw_subdir_name.

    raw_subdir_name:
        Name of the subdirectory containing navigation-processed raw files.

    single_raw_file:
        Optional file to process. If this is a filename only, it is interpreted
        relative to cruise_dir / raw_subdir_name. If None, all files matching
        raw_glob are processed.

    raw_glob:
        File pattern used when single_raw_file is None.

    output_subdir_name:
        Output directory name under cruise_dir for combined depth CSVs.
        The default is "depth" to match the current project convention.

    image_subdir_name:
        Output directory name under cruise_dir for QC images.

    error_log_name:
        Error log filename written directly under cruise_dir.

    export_bottom_depth_csv:
        If True, write the combined bottom-depth CSV for each raw file.

    save_depth_images:
        If True, save one echogram/QC image per raw file and frequency/channel.

    run_blackwell_detector:
        If True, also run the Blackwell detector when split-beam angle data are
        available. Basic detection always runs.

    show_plots:
        If True, show plots interactively. For batch runs this should usually
        remain False.
    """

    cruise_dir: Path | str
    raw_subdir_name: str = "raw_w_nav"
    single_raw_file: Path | str | None = None
    raw_glob: str = "*.raw"
    output_subdir_name: str = "depth"
    image_subdir_name: str = "images"
    error_log_name: str = "depth_processing_errors.txt"
    export_bottom_depth_csv: bool = True
    save_depth_images: bool = True
    run_blackwell_detector: bool = False
    depth_image_format: str = "png"
    depth_image_dpi: int = 200
    show_plots: bool = False
    plot_sv: bool = False
    plot_basic_seafloor: bool = False
    plot_blackwell_depth: bool = False
    plot_blackwell_angles: bool = False
    plot_compare_bottom_detections: bool = False
    n_skip: int = 20

    # Basic detector settings.
    basic_threshold: tuple[float, float] = (-40, -20)
    basic_offset_m: float = 0.3

    # Blackwell detector settings.
    blackwell_threshold: list[float] | tuple[float, float, float] = (-40, 702, 282)
    blackwell_offset: float = 0.3
    blackwell_r0: float = 10
    blackwell_r1: float = 1000
    blackwell_wtheta: float = 28
    blackwell_wphi: float = 52

    @property
    def cruise_path(self) -> Path:
        """Return cruise_dir as an expanded Path."""
        return Path(self.cruise_dir).expanduser()

    @property
    def raw_dir(self) -> Path:
        """Return the directory containing navigation-processed raw files."""
        return self.cruise_path / self.raw_subdir_name

    @property
    def output_dir(self) -> Path:
        """Return the combined depth CSV output directory."""
        return self.cruise_path / self.output_subdir_name

    @property
    def image_dir(self) -> Path:
        """Return the QC image output directory."""
        return self.cruise_path / self.image_subdir_name

    @property
    def error_log(self) -> Path:
        """Return the cruise-level error log path."""
        return self.cruise_path / self.error_log_name


# =============================================================================
# General path and reporting helpers
# =============================================================================


def relpath(path: Path) -> str:
    """Return a relative path for cleaner console output when possible."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return os.path.relpath(str(path), start=str(Path.cwd()))


def get_output_dirs(config: DepthExtractionConfig) -> tuple[Path, Path, Path]:
    """Return output, image, and error-log paths and create needed directories."""
    output_dir = config.output_dir
    image_dir = config.image_dir
    error_log = config.error_log

    output_dir.mkdir(parents=True, exist_ok=True)

    if config.save_depth_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, image_dir, error_log


def discover_raw_files(config: DepthExtractionConfig) -> list[Path]:
    """Return the raw files to process based on the supplied configuration."""
    if config.single_raw_file is not None:
        single_raw_file = Path(config.single_raw_file).expanduser()

        # If only a filename was supplied, interpret it relative to raw_dir.
        if not single_raw_file.is_absolute() and single_raw_file.parent == Path("."):
            single_raw_file = config.raw_dir / single_raw_file

        return [single_raw_file]

    return sorted(config.raw_dir.glob(config.raw_glob))


def append_error(error_log: Path, raw_file: Path, error: BaseException) -> None:
    """Append a failed filename and traceback to the cruise-level error log."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    error_log.parent.mkdir(parents=True, exist_ok=True)

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
    text = str(value).strip()
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


def output_combined_csv_path(raw_file: Path, output_dir: Path) -> Path:
    """Build the one-file-per-raw-file combined depth CSV path."""
    return output_dir / f"{raw_file.stem}_depth.csv"


def output_image_path(
    raw_file: Path,
    image_dir: Path,
    frequency_hz: float,
    config: DepthExtractionConfig,
) -> Path:
    """Build the per-file/per-channel QC image path."""
    freq_label = frequency_to_short_label(frequency_hz)
    ext = config.depth_image_format.lower().lstrip(".")
    return image_dir / f"{raw_file.stem}_{freq_label}_bottom_detection.{ext}"


# =============================================================================
# Data preparation helpers
# =============================================================================


def get_sound_speed_for_channel(ed: object, channel: object) -> xr.DataArray | None:
    """Return channel-specific sound speed aligned to ping_time, if available.

    For EK60/ES60 files read by Echopype, sound speed is commonly stored in
    ed["Environment"] as sound_speed_indicative with dimensions such as
    (channel, time1). Bottom detections and navigation use ping_time, so this
    helper selects the requested channel and renames the environment time
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

        if "channel" in ss.dims:
            ss = ss.sel(channel=channel)

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


def run_basic_detector(
    ds_sv: xr.Dataset,
    channel: object,
    config: DepthExtractionConfig,
) -> xr.DataArray:
    """Run Echopype's basic seafloor detector for one channel."""
    depth = detect_seafloor(
        ds_sv,
        method="basic",
        params={
            "var_name": "Sv",
            "channel": channel,
            "threshold": config.basic_threshold,
            "offset_m": config.basic_offset_m,
            "bin_skip_from_surface": config.n_skip,
        },
    )

    if not isinstance(depth, xr.DataArray):
        raise TypeError("Basic detector did not return an xarray.DataArray.")

    if set(depth.dims) != {"ping_time"}:
        raise ValueError(f"Unexpected basic depth dimensions: {depth.dims}")

    return depth


def run_blackwell_detector(
    ds_sv: xr.Dataset,
    channel: object,
    config: DepthExtractionConfig,
) -> xr.DataArray:
    """Run Echopype's Blackwell seafloor detector for one channel."""
    depth = detect_seafloor(
        ds=ds_sv,
        method="blackwell",
        params={
            "channel": channel,
            "var_name": "Sv",
            "threshold": list(config.blackwell_threshold),
            "offset": config.blackwell_offset,
            "r0": config.blackwell_r0,
            "r1": config.blackwell_r1,
            "wtheta": config.blackwell_wtheta,
            "wphi": config.blackwell_wphi,
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
    config: DepthExtractionConfig,
) -> dict[object, dict[str, xr.DataArray]]:
    """Run all enabled bottom-detection methods for each available channel."""
    detections: dict[object, dict[str, xr.DataArray]] = {}

    for channel in channels_available:
        print(f"\nRunning bottom detection for: {channel}")

        detections[channel] = {}
        detections[channel]["basic"] = run_basic_detector(ds_sv, channel, config)

        if config.run_blackwell_detector and has_splitbeam_angles:
            detections[channel]["blackwell"] = run_blackwell_detector(
                ds_sv,
                channel,
                config,
            )
        elif config.run_blackwell_detector and not has_splitbeam_angles:
            print("  Skipping Blackwell detector because split-beam angles are missing.")

    return detections


# =============================================================================
# CSV export helpers
# =============================================================================


def make_navigation_dataframe(lat: xr.DataArray, lon: xr.DataArray) -> pd.DataFrame:
    """Build the base navigation table used by the combined CSV."""
    lat_aligned, lon_aligned = xr.align(lat, lon, join="inner")

    if not lat_aligned.ping_time.equals(lon_aligned.ping_time):
        raise ValueError("Latitude and longitude ping times are not aligned.")

    return pd.DataFrame(
        {
            "ping_time": lat_aligned["ping_time"].values,
            "latitude": np.round(lat_aligned.values, 6),
            "longitude": np.round(lon_aligned.values, 6),
        }
    )


def make_depth_column_dataframe(depth: xr.DataArray, column_name: str) -> pd.DataFrame:
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
        print(f"Cannot export {column_name}; sound speed has no ping_time dimension.")
        return None

    return pd.DataFrame(
        {
            "ping_time": sound_speed["ping_time"].values,
            column_name: np.round(ss_values, 2),
        }
    )


def merge_on_ping_time(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer-merge two tables on ping_time without dropping unmatched pings."""
    return left.merge(right, on="ping_time", how="outer")


def export_combined_bottom_depth_csv(
    raw_file: Path,
    output_dir: Path,
    ed: object,
    ds_sv: xr.Dataset,
    detections: dict[object, dict[str, xr.DataArray]],
    config: DepthExtractionConfig,
) -> Path | None:
    """Export one combined bottom-depth CSV per raw file."""
    if not config.export_bottom_depth_csv:
        return None

    if not check_navigation(ds_sv):
        print("Cannot export bottom-depth CSV because navigation is missing.")
        return None

    lat = ds_sv["latitude"]
    lon = ds_sv["longitude"]
    frequency_nominal = ds_sv["frequency_nominal"]

    combined_df = make_navigation_dataframe(lat, lon)

    for channel, method_depths in detections.items():
        frequency_hz = float(frequency_nominal.sel(channel=channel).values)
        freq_label = frequency_to_short_label(frequency_hz)

        sound_speed_channel = get_sound_speed_for_channel(ed=ed, channel=channel)

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
            depth_df = make_depth_column_dataframe(depth=depth, column_name=depth_col)
            combined_df = merge_on_ping_time(combined_df, depth_df)

    combined_df = combined_df.sort_values("ping_time")

    # Re-round after merges so the exported CSV has the intended precision.
    if "latitude" in combined_df:
        combined_df["latitude"] = combined_df["latitude"].round(6)
    if "longitude" in combined_df:
        combined_df["longitude"] = combined_df["longitude"].round(6)

    for col in combined_df.columns:
        if (
            col.endswith("_sound_speed_m_s")
            or col.endswith("_basic")
            or col.endswith("_blackwell")
        ):
            combined_df[col] = combined_df[col].round(2)

    csv_path = output_combined_csv_path(raw_file=raw_file, output_dir=output_dir)
    combined_df.to_csv(csv_path, index=False)

    print(f"Saved combined bottom-depth CSV to: {relpath(csv_path)}")
    return csv_path


# =============================================================================
# Plotting and image helpers
# =============================================================================


def maybe_show_or_close(fig: plt.Figure, config: DepthExtractionConfig) -> None:
    """Show a figure interactively or close it to avoid memory buildup."""
    if config.show_plots:
        plt.show()
    else:
        plt.close(fig)


def plot_sv_diagnostics(ds_sv: xr.Dataset, n_channels: int, config: DepthExtractionConfig) -> None:
    """Optional exploratory Sv plots from the original standalone script."""
    if not config.plot_sv:
        return

    plot_obj = ds_sv["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-80,
        vmax=-30,
        cmap="RdYlBu_r",
        yincrease=False,
    )
    maybe_show_or_close(plot_obj.fig, config)

    plot_obj = ds_sv["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-40,
        vmax=-20,
        cmap="RdYlBu_r",
        yincrease=False,
    )
    maybe_show_or_close(plot_obj.fig, config)

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
    maybe_show_or_close(plot_obj.fig, config)


def plot_simple_depth_diagnostics(
    detections: dict[object, dict[str, xr.DataArray]],
    config: DepthExtractionConfig,
) -> None:
    """Optional simple line/point plots for bottom detections."""
    for channel, method_depths in detections.items():
        if config.plot_basic_seafloor and "basic" in method_depths:
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
            maybe_show_or_close(fig, config)

        if config.plot_blackwell_depth and "blackwell" in method_depths:
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
            maybe_show_or_close(fig, config)


def plot_blackwell_angle_diagnostics(
    channels_available: np.ndarray,
    angle_along: xr.DataArray | None,
    angle_athwart: xr.DataArray | None,
    config: DepthExtractionConfig,
) -> None:
    """Optional split-beam angle plots from the original standalone script."""
    if not config.plot_blackwell_angles:
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
        maybe_show_or_close(fig, config)


def plot_compare_bottom_detections(
    detections: dict[object, dict[str, xr.DataArray]],
    config: DepthExtractionConfig,
) -> None:
    """Optional comparison plot for basic and Blackwell detections."""
    if not config.plot_compare_bottom_detections:
        return

    for channel, method_depths in detections.items():
        if "basic" not in method_depths or "blackwell" not in method_depths:
            continue

        basic_depth = method_depths["basic"]
        blackwell_depth = method_depths["blackwell"]

        bd, bw = xr.align(basic_depth, blackwell_depth, join="inner")

        if not bd.ping_time.equals(bw.ping_time):
            raise ValueError("Basic and Blackwell ping times are not aligned.")

        diff = bd - bw
        common_time = bd.ping_time

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharex=True)

        axs[0].plot(common_time, bd, ".", label="Basic", color="navy", markersize=1)
        axs[0].plot(common_time, bw, ".", label="Blackwell", color="orange", markersize=1)
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
        maybe_show_or_close(fig, config)


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
    config: DepthExtractionConfig,
) -> list[Path]:
    """Save one QC echogram image per input file and channel."""
    saved_paths: list[Path] = []

    if not config.save_depth_images:
        return saved_paths

    frequency_nominal = ds_sv["frequency_nominal"]

    for channel, method_depths in detections.items():
        frequency_hz = float(frequency_nominal.sel(channel=channel).values)
        fig = make_echogram_bottom_figure(ds_sv, channel, method_depths)
        image_path = output_image_path(raw_file, image_dir, frequency_hz, config)

        fig.savefig(
            image_path,
            dpi=config.depth_image_dpi,
            bbox_inches="tight",
        )
        saved_paths.append(image_path)
        print(f"Saved depth QC image to: {relpath(image_path)}")
        maybe_show_or_close(fig, config)

    return saved_paths


# =============================================================================
# Per-file and cruise-level processing
# =============================================================================


def process_raw_file(
    raw_file: Path,
    output_dir: Path,
    image_dir: Path,
    config: DepthExtractionConfig,
) -> dict[str, object]:
    """Process one EK60/ES60 .raw file and write requested outputs."""
    print("\n" + "=" * 80)
    print(f"Processing raw file: {relpath(raw_file)}")

    ed = ep.open_raw(raw_file, sonar_model="EK60")

    beam_group = ed["Sonar/Beam_group1"]
    channels_available = beam_group["channel"].values
    n_channels = len(channels_available)

    print("Available channels:")
    for channel in channels_available:
        print(f"  {channel}")

    ds_sv = ep.calibrate.compute_Sv(ed)

    # Add depth coordinates relative to the transducer.
    ds_sv = ep.consolidate.add_depth(ds_sv, ed, depth_offset=0)

    # Add position information from parsed NMEA/navigation records.
    ds_sv = ep.consolidate.add_location(ds_sv, ed)

    plot_sv_diagnostics(ds_sv, n_channels, config)

    ds_sv, has_splitbeam_angles, angle_along, angle_athwart = (
        add_splitbeam_angles_if_available(ds_sv, beam_group)
    )

    detections = run_bottom_detection(
        ds_sv=ds_sv,
        channels_available=channels_available,
        has_splitbeam_angles=has_splitbeam_angles,
        config=config,
    )

    plot_simple_depth_diagnostics(detections, config)
    plot_blackwell_angle_diagnostics(
        channels_available,
        angle_along,
        angle_athwart,
        config,
    )
    plot_compare_bottom_detections(detections, config)

    csv_path = export_combined_bottom_depth_csv(
        raw_file=raw_file,
        output_dir=output_dir,
        ed=ed,
        ds_sv=ds_sv,
        detections=detections,
        config=config,
    )
    image_paths = save_depth_images(raw_file, image_dir, ds_sv, detections, config)

    print(f"Finished processing: {raw_file.name}")

    return {
        "file": raw_file.name,
        "source_path": str(raw_file),
        "success": True,
        "csv_path": str(csv_path) if csv_path is not None else "",
        "image_count": len(image_paths),
        "image_paths": ";".join(str(path) for path in image_paths),
        "n_channels": int(n_channels),
        "channels": ";".join(str(channel) for channel in channels_available),
        "has_splitbeam_angles": bool(has_splitbeam_angles),
        "run_blackwell_detector": config.run_blackwell_detector,
        "error": "",
    }


def process_cruise(config: DepthExtractionConfig) -> pd.DataFrame:
    """Process all requested raw files and keep going if individual files fail.

    Returns
    -------
    pandas.DataFrame
        One summary row per requested raw file. Failed files are included with
        success=False and the error message recorded.
    """
    output_dir, image_dir, error_log = get_output_dirs(config)
    raw_files = discover_raw_files(config)

    if not raw_files:
        raise FileNotFoundError(f"No raw files found in RAW_DIR: {config.raw_dir}")

    print(f"Found {len(raw_files)} raw file(s) to process.")
    print(f"Raw input directory: {relpath(config.raw_dir)}")
    print(f"CSV output directory: {relpath(output_dir)}")

    if config.save_depth_images:
        print(f"Image output directory: {relpath(image_dir)}")

    print(f"Error log: {relpath(error_log)}")

    results: list[dict[str, object]] = []

    for raw_file in raw_files:
        try:
            result = process_raw_file(raw_file, output_dir, image_dir, config)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - batch processing should continue.
            print(f"ERROR: Failed processing {relpath(raw_file)}")
            print(f"  {type(exc).__name__}: {exc}")
            append_error(error_log, raw_file, exc)
            results.append(
                {
                    "file": raw_file.name,
                    "source_path": str(raw_file),
                    "success": False,
                    "csv_path": "",
                    "image_count": 0,
                    "image_paths": "",
                    "n_channels": 0,
                    "channels": "",
                    "has_splitbeam_angles": False,
                    "run_blackwell_detector": config.run_blackwell_detector,
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(results)

    n_success = int(summary_df["success"].sum()) if not summary_df.empty else 0
    n_failed = int((~summary_df["success"]).sum()) if not summary_df.empty else 0

    print("\n" + "=" * 80)
    print("Processing complete.")
    print(f"Successful files: {n_success}")
    print(f"Failed files:     {n_failed}")

    if n_failed:
        print(f"See error log: {relpath(error_log)}")

    return summary_df


# Alias that mirrors the nav_checker.py pattern.
def main(config: DepthExtractionConfig) -> pd.DataFrame:
    """Run bottom-depth extraction using a DepthExtractionConfig."""
    return process_cruise(config)
