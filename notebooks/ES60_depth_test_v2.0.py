from pathlib import Path

import echopype as ep
from echopype.mask import detect_seafloor
import hvplot.xarray  # noqa: F401  # Enables interactive xarray/hvplot support.
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# Adapted from:
# https://echopype-examples.readthedocs.io/en/latest/seafloor_detection.html


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

PLOT_SV = True
PLOT_BASIC_SEAFLOOR = True
PLOT_BLACKWELL_DEPTH = True
PLOT_BLACKWELL_ANGLES = True
PLOT_COMPARE_BOTTOM_DETECTIONS = True
PLOT_BOTTOM_ON_ECHOGRAM = True

N_SKIP = 20

RAW_PATH = Path(
    "../CRUISES_processed/EBS_2017_AlaskaKnight_leg_4/"
    "raw_w_nav/L0531-D20170904-T100303-ES60.raw"
)


# -----------------------------------------------------------------------------
# Basic bottom-detection settings
# -----------------------------------------------------------------------------

BASIC_THRESHOLD = (-40, -20)
BASIC_OFFSET_M = 0.3


# -----------------------------------------------------------------------------
# Blackwell bottom-detection settings
# -----------------------------------------------------------------------------

BLACKWELL_THRESHOLD = [-40, 702, 282]
BLACKWELL_OFFSET = 0.3
BLACKWELL_R0 = 10
BLACKWELL_R1 = 1000
BLACKWELL_WTHETA = 28
BLACKWELL_WPHI = 52


# -----------------------------------------------------------------------------
# Read raw EK60 file
# -----------------------------------------------------------------------------

ed = ep.open_raw(RAW_PATH, sonar_model="EK60")

beam_group = ed["Sonar/Beam_group1"]
channels_available = beam_group["channel"].values
n_channels = len(channels_available)

print("Available channels:")
for channel in channels_available:
    print(f"  {channel}")


# -----------------------------------------------------------------------------
# Compute Sv and add depth/location
# -----------------------------------------------------------------------------

ds_Sv = ep.calibrate.compute_Sv(ed)

# Add depth coordinates relative to the transducer.
ds_Sv = ep.consolidate.add_depth(ds_Sv, ed, depth_offset=0)

# Add position information from the parsed NMEA/navigation records.
ds_Sv = ep.consolidate.add_location(ds_Sv, ed)


# -----------------------------------------------------------------------------
# Plot Sv
# -----------------------------------------------------------------------------

if PLOT_SV:
    # Full Sv range.
    ds_Sv["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-80,
        vmax=-30,
        cmap="RdYlBu_r",
        yincrease=False,
    )
    plt.show()

    # Narrower Sv range to emphasize the seafloor return.
    ds_Sv["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-40,
        vmax=-20,
        cmap="RdYlBu_r",
        yincrease=False,
    )
    plt.show()

    # First range samples only, to inspect the surface saturation zone.
    ds_Sv.isel(range_sample=slice(0, 50))["Sv"].plot(
        x="ping_time",
        row="channel",
        col_wrap=n_channels,
        vmin=-40,
        vmax=-20,
        cmap="RdYlBu_r",
        yincrease=False,
        ylim=(50, 0),
    )
    plt.show()


# -----------------------------------------------------------------------------
# Basic bottom detection
# -----------------------------------------------------------------------------

# For now, work on the first channel only.
# Later this can become a loop over channels.
sel_channel = channels_available[0]

basic_depth = detect_seafloor(
    ds_Sv,
    method="basic",
    params={
        "var_name": "Sv",
        "channel": sel_channel,
        "threshold": BASIC_THRESHOLD,
        "offset_m": BASIC_OFFSET_M,
        "bin_skip_from_surface": N_SKIP,
    },
)

# Sanity checks.
assert isinstance(basic_depth, xr.DataArray)
assert set(basic_depth.dims) == {"ping_time"}

if PLOT_BASIC_SEAFLOOR:
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        basic_depth["ping_time"].values,
        basic_depth.values,
        ".",
        markersize=1,
    )

    ax.set_title(f"Basic seafloor depth: {sel_channel}")
    ax.set_xlabel("Ping time")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()

    plt.show()


# -----------------------------------------------------------------------------
# Blackwell bottom detection
# -----------------------------------------------------------------------------

# The Blackwell method requires split-beam angle data.
required_vars = [
    "angle_alongship",
    "angle_athwartship",
]

missing_vars = [
    var for var in required_vars
    if var not in beam_group
]

HAS_SPLITBEAM_ANGLES = len(missing_vars) == 0

if not HAS_SPLITBEAM_ANGLES:
    print("Missing required split-beam variables:")
    for var in missing_vars:
        print(f"  {var}")

else:
    print("Split-beam angle variables found.")

    # Extract split-beam angle variables.
    angle_along = beam_group["angle_alongship"]
    angle_athwart = beam_group["angle_athwartship"]

    # Merge into Sv dataset.
    ds_Sv = ds_Sv.assign(
        angle_alongship=angle_along,
        angle_athwartship=angle_athwart,
    )

    blackwell_depth = detect_seafloor(
        ds=ds_Sv,
        method="blackwell",
        params={
            "channel": sel_channel,
            "var_name": "Sv",
            "threshold": BLACKWELL_THRESHOLD,
            "offset": BLACKWELL_OFFSET,
            "r0": BLACKWELL_R0,
            "r1": BLACKWELL_R1,
            "wtheta": BLACKWELL_WTHETA,
            "wphi": BLACKWELL_WPHI,
        },
    )

    # Sanity checks.
    assert isinstance(blackwell_depth, xr.DataArray)
    assert set(blackwell_depth.dims) == {"ping_time"}

    if PLOT_BLACKWELL_DEPTH:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            blackwell_depth["ping_time"].values,
            blackwell_depth.values,
            ".",
            label="Blackwell",
            color="firebrick",
            markersize=1,
        )

        ax.invert_yaxis()
        ax.set_title(f"Blackwell seafloor depth: {sel_channel}")
        ax.set_xlabel("Ping time")
        ax.set_ylabel("Depth (m)")
        ax.legend()
        ax.grid(True)

        plt.show()

    if PLOT_BLACKWELL_ANGLES:
        angle_athwart_sel = angle_athwart.sel(channel=sel_channel)
        angle_along_sel = angle_along.sel(channel=sel_channel)

        fig, axes = plt.subplots(
            ncols=2,
            figsize=(14, 5),
            sharey=True,
        )

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

        fig.suptitle(f"Channel: {sel_channel}")

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # Compare basic vs. Blackwell bottom detections
    # -------------------------------------------------------------------------

    if PLOT_COMPARE_BOTTOM_DETECTIONS:
        pt_basic = basic_depth.ping_time
        pt_blackwell = blackwell_depth.ping_time

        missing_in_blackwell = pt_basic[~pt_basic.isin(pt_blackwell)]
        missing_in_basic = pt_blackwell[~pt_blackwell.isin(pt_basic)]

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

        bd, bw = xr.align(
            basic_depth,
            blackwell_depth,
            join="inner",
        )

        assert bd.ping_time.equals(bw.ping_time), (
            "Ping times are not aligned."
        )

        diff = bd - bw
        common_time = bd.ping_time

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(14, 5),
            sharex=True,
        )

        axs[0].plot(
            common_time,
            bd,
            ".",
            label="Basic",
            color="navy",
            markersize=1,
        )

        axs[0].plot(
            common_time,
            bw,
            ".",
            label="Blackwell",
            color="firebrick",
            markersize=1,
        )

        axs[0].invert_yaxis()
        axs[0].set_title("Seafloor depth over ping time")
        axs[0].set_xlabel("Ping time")
        axs[0].set_ylabel("Depth (m)")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(
            common_time,
            diff,
            ".",
            color="darkgreen",
            markersize=1,
        )

        axs[1].axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

        axs[1].set_title("Difference: Basic – Blackwell")
        axs[1].set_xlabel("Ping time")
        axs[1].set_ylabel("Depth difference (m)")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# QC plot: echogram with detected seafloor line
# -----------------------------------------------------------------------------

if PLOT_BOTTOM_ON_ECHOGRAM:
    selected_channel = sel_channel

    # Extract Sv and depth for one channel.
    Sv_da = ds_Sv["Sv"].sel(channel=selected_channel)
    depth = ds_Sv["depth"].sel(channel=selected_channel).isel(ping_time=0)

    # Remove non-finite depth values before building the depth-coordinate array.
    finite_depth = np.isfinite(depth.values)

    Sv_da_finite = Sv_da.isel(range_sample=finite_depth)
    depth_finite = depth.values[finite_depth]

    # Sort by depth so pcolormesh gets an ordered y-axis.
    sort_idx = np.argsort(depth_finite)

    Sv_values_sorted = Sv_da_finite.values[:, sort_idx]
    depth_sorted = depth_finite[sort_idx]

    # Build DataArray with explicit depth coordinate.
    Sv_plot = xr.DataArray(
        data=Sv_values_sorted,
        dims=["ping_time", "depth"],
        coords={
            "ping_time": Sv_da["ping_time"].values,
            "depth": depth_sorted,
        },
        name="Sv",
    ).expand_dims(channel=[selected_channel])

    # Wrap into Dataset.
    ds_single = xr.Dataset(
        {
            "Sv": Sv_plot,
        }
    )

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

    ax.plot(
        basic_depth["ping_time"].values,
        basic_depth.values,
        color="cyan",
        label="Basic bottom",
        linewidth=1.0,
    )

    if HAS_SPLITBEAM_ANGLES:
        ax.plot(
            blackwell_depth["ping_time"].values,
            blackwell_depth.values,
            color="black",
            label="Blackwell bottom",
            linewidth=1.5,
        )

    ax.set_title(f"Echogram and detected seafloor: {selected_channel}")
    ax.set_xlabel("Ping time")
    ax.set_ylabel("Depth (m)")
    ax.legend()

    plt.tight_layout()
    plt.show()


