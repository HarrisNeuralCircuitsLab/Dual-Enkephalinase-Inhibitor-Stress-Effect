# Project Title: Dual Enkephalinase Inhibitor Effects on Stress-Induced Reward Seeking Neural and Behavioral Deficits

## Description

This repository contains the code and analysis notebooks for the project investigating the effects of a dual enkephalinase inhibitor on stress-induced behavioral and neural deficits in mice. The analysis focuses on licking behavior as a proxy for motivational and reward-seeking behavior, as well as fiber photometry data for neural activity.

The code that was developed to conduct the experiments and analyse the data, was made available at: https://github.com/HarrisNeuralCircuitsLab/Dual-Enkephalinase-Inhibitor-Stress-Effect

## Repository Structure

- `Behavioral Analyses/`: Contains Jupyter Notebooks for behavioral data analysis for each animal.
  - `modules/`: Python modules with helper functions for behavioral data loading and session analysis.
- `Neural Analyses/`: Contains Jupyter Notebooks for fiber photometry data analysis.
  - `modules/`: Python modules with helper functions for fiber photometry data processing and analysis.
- `data/`: 

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab
- Required Python packages can be installed via pip:
  ```bash
  pip install numpy pandas matplotlib scipy
  ```
## Data Availability

The data collected at this phase are available in NWB format at: https://dandiarchive.org/dandiset/001564

This code can be applied directly to the unformatted data uploaded at https://zenodo.org/uploads/16967794

### Usage

1.  **Clone the repository:**
    ```bash
    git clone [URL of this repository]
    ```
2.  **Place the data:**
    Download the data and place the animal session folders into a `data/` directory at the root of the project.
3.  **Update data path:**
    In the notebooks, update the `DATA_DIR` variable to point to the `data/` directory.
4.  **Run the analysis:**
    Open and run the cells in the notebooks inside `Behavioral Analyses/` and `Neural Analyses/` to reproduce the analysis and figures.
