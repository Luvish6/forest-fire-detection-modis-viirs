# Data Sources

This directory is intended for the storage of dataset files. Due to GitHub's file size limits and copyright/distribution policies, the raw data files are not included in this repository.

## Required Datasets

1.  **MODIS Active Fire Data** (CSV)
2.  **VIIRS-SNPP Active Fire Data** (CSV)

## Download Instructions

You can download the latest fire data from **NASA FIRMS (Fire Information for Resource Management System)**:

*   **URL**: [https://firms.modaps.eosdis.nasa.gov/download/](https://firms.modaps.eosdis.nasa.gov/download/)
*   **Format**: Select "CSV" format.
*   **Region**: South Asia (or your region of interest).
*   **Data Source**:
    *   For MODIS: Select `MODIS C6.1`.
    *   For VIIRS: Select `VIIRS S-NPP`.

## Setup

1.  Download the CSV files.
2.  Rename them to standard names for easier usage (optional, but recommended):
    *   `modis_2024.csv`
    *   `viirs_2024.csv`
3.  Place them in this `data/` directory.

## Expected Columns

The scripts expect standard FIRMS columns including:
*   `latitude`, `longitude`, `brightness`, `scan`, `track`, `acq_date`, `acq_time`, `satellite`, `instrument`, `confidence`, `version`, `bright_t31`, `frp`, `daynight`.
