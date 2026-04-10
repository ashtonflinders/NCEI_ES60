# NCEI_ES60

## Overview
**NCEI_ES60** provides Python tools for organizing and processing Simrad **ES60** echosounder `.raw` data using the `echopype` library. The repository supports data quality assessment, navigation validation, and future extraction of bottom depths for bathymetric and hydroacoustic analyses.

> **Note:** Although `echopype` uses the `"EK60"` sonar model, it is fully compatible with Simrad **ES60** data because both systems share the same `.raw` file format.

---

## Dependencies and Installation

This project depends on the **echopype** library for reading and processing echosounder data.

### Install echopype (Development Mode)

The recommended setup uses **Python 3.11** and installs `echopype` in development mode:

```bash
# Create and activate a conda environment
conda create --name echopype python=3.11
conda activate echopype

# Clone the echopype repository
git clone https://github.com/OSOceanAcoustics/echopype.git
cd echopype

# Install echopype with development dependencies
pip install -e ".[dev]"
```

### echopype Resources
- **GitHub Repository:** https://github.com/OSOceanAcoustics/echopype
- **Documentation:** https://echopype.readthedocs.io/

### Install NCEI_ES60

After installing `echopype`, clone this repository:

```bash
git clone https://github.com/ashtonflinders/NCEI_ES60.git
cd NCEI_ES60
```

Additional required Python packages:

```bash
pip install numpy pandas
```

---

## nav_checker.py

### Summary
`nav_checker.py` is a utility for assessing the presence of usable navigation information in Simrad ES60 `.raw` files and organizing datasets into a standardized directory structure. The script analyzes **top-level** `.raw` files within a user-specified directory using `echopype` to determine whether valid latitude and longitude data—and optionally NMEA sentence metadata—are present.

### Key Features
- **Navigation Validation:** Detects usable navigation data based on latitude and longitude, with optional inclusion of NMEA sentence metadata.
- **Standardized File Organization:** Optionally organizes files into the following directories:
  - `raw_w_nav` – Files containing usable navigation data.
  - `raw_no_nav` – Files lacking usable navigation data.
  - `raw_error` – Files that could not be processed.
  - `calibration` – Directories identified as calibration data.
  - `other` – Non-`.raw` files.
- **Summary Reporting:** Generates a `raw_nav_summary.csv` file summarizing navigation availability and diagnostic metrics for each processed file.
- **Read-Only Mode:** Allows users to evaluate datasets and generate the summary report without modifying the existing directory structure or moving files.
- **Top-Level Processing:** Only `.raw` files in the specified directory are processed, ensuring that calibration data in subdirectories remain unaffected.
- **Clean Console Output:** Displays file paths relative to the current working directory for improved readability.

### Usage

#### Command Line
```bash
python nav_checker.py /path/to/source_directory
```

#### Interactive Python Session
```python
import nav_checker as nc

# Run in read-only mode (no file movement)
nc.READ_ONLY_MODE = True
nc.MOVE_FILES = False # Not used if in Read-only mode
nc.USE_SENTENCE_FOR_NAV = True # Additional check for NMEA string
summary_df = nc.main("/path/to/source_directory")
print(summary_df.head())
```

### Example Output Structure
```
source_directory/
├── raw_w_nav/
├── raw_no_nav/
├── raw_error/
├── calibration/
├── other/
└── raw_nav_summary.csv
```

---

## Repository Structure
```
NCEI_ES60/
├── nav_checker.py
├── README.md
├── .gitignore
├── CRUISES/        # Cruise data (local only, not synced to GitHub)
├── cruise_nav_summaries/ 
├── notes/
```

---

## Future Development
Planned enhancements to this repository include:
- Correct for possible embedded triangular wave error (see `notes` folder)
- Extraction of **bottom depths** from ES60 `.raw` files using `echopype`.
- Additional quality control and visualization tools.
- Support for automated processing workflows.

---

## Author
**Ashton Flinders**  
Email: andrealphus@gmail.com

---

## License
This project is released under the **MIT License**. See the `LICENSE` file for details.
