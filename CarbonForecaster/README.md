# Predicting Corporate Carbon Emissions

## Introduction

Welcome to my MSc thesis repository. This contains the codebase I have built, **CarbonForecaster**, to model firms' Scope 3 emissions using data from the London Stock Exchange Group (LSEG). Naturally, there is no proprietary data or reference to individual firms contained herein.

The structure of the workflow is as follows.

![MLWorkFlow drawio (1)](https://github.com/user-attachments/assets/449f6162-37e2-4a59-af2a-9449e1119eb8)

## Repository Structure
```
CarbonForecaster/ 
│ ├── data/                               # Directory for data; only contains publicly-sourced data in the repo 
  ├── presentations/                      # Directory for PowerPoint presentations 
  ├── report/                             # Directory for all files relating to the thesis write-up (hidden from public) 
  ├── src/                                # Directory containing the key source code for the project 
  │ └── figures/                          # Subdirectory for some generic charting functions 
  │ └── models/                           # Subdirectory for holding all pickled model objects 
  │ └── tuned/                            # Subsubdirectory for holding all tuned models, named according to consistent pattern 
  │ └── tuning/                           # Subsubdirectory for holding all scripts for tuning against target variables 
  │ └── tuning.py                         # A generic tuning script to demonstrate how tuning takes place 
  │ └── api_excel_ftse350_comparison.py   # Script to check if the Python API returns identical results to Excel plug-in 
  │ └── charts.py                         # Plotting code for most charts in the report (which don't require heavy data work) 
  │ └── clean_raw_data.py                 # Custom functions for cleaning the raw data 
  │ └── eda.py                            # Helper functions for exploring the data 
  │ └── extension_1_outliers.py           # Code for Section 6.1 of the report 
  │ └── extension_2_underreporting.py     # Code for Section 6.2 of the report 
  │ └── extension_3_outliers.py           # Code for Section 6.3 of the report 
  │ └── load_raw_data.py                  # Script to pull data via the Python API 
  │ └── ml_pipeline.py                    # Key functions for what the machine learning pipeline requires 
  │ └── most_material_s3_categories.py    # Materiality code 
  │ └── preprocessing.py                  # Functions for preprocessing the data ahead of modeling 
  │ └── stepwise_regression.py            # Quick exercise for seeing which variables are included in stepwise regression 
  │ └── stratified_sampling.py            # Code for creating appropriate peer groups given k folds and n firms per fold 
  ├── tests/                              # Directory containing a sample of tests for custom functions 
  ├── LICENSE                             # Standard MIT license 
  ├── README.md                           # You're reading it! 
  ├── pyproject.toml                      # Toml used to build the package 
  ├── requirements.txt                    # Python dependencies required to run the project
```


## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Michael-Hollins/Imperial_MSc_AI.git
   cd Imperial_MSc_AI/CarbonForecaster
   pip install -r requirements.txt
