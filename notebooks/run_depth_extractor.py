"""
Example runner for ncei_es60.depth_extractor.

Edit the settings below, then run:

    python run_depth_extractor.py

Or copy the same pattern into a notebook cell.
"""

from pathlib import Path

from ncei_es60.depth_extractor import DepthExtractionConfig, process_cruise


config = DepthExtractionConfig(
    cruise_dir=Path("../CRUISES_processed/EBS17VA_example"),
    single_raw_file="L0004-D20170602-T224458-ES60.raw",  # Use None for all *.raw files in raw_w_nav.
    output_subdir_name="depth",
    image_subdir_name="images",
    run_blackwell_detector=True,
    save_depth_images=True,
    show_plots = True
)

summary_df = process_cruise(config)
print("\nSummary table:")
print(summary_df)
