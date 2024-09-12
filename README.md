# ToxDTIEngine

## Overview

This repository, **ToxDTIEngine**, contains the code and associated resources for the research presented in the manuscript titled "**Evaluating the Contribution of Drug-Binding Targets to Side Effect Prediction in Drug Development**". The research focuses on predicting drug-induced side effects using protein-protein interactions (PPIs) by incorporating drug-target interaction (DTI) data from multiple databases. **ToxDTIEngine** uses another tool called [**PathFX**](https://github.com/jenwilson521/PathFX), which is a PPI tool, and builds a DTI pipeline employing predictive modeling to select drug targets and rank databases in predicting side effects.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Citation](#citation)

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes. See [Usage](#usage) for notes on how to use the code in a research context.

## Prerequisites

Ensure you have the following prerequisites to run the code:

- **ToxDTIEngine** was developed using the Python programming language.
- Python 3.x must be installed, along with the following Python libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn

## Usage

Detailed instructions on how to use the code:

- Clone this repository to your local machine or cluster:
  ```bash
  git clone https://github.com/yourusername/ToxDTIEngine.git

- The data/ folder contains both raw and processed drug-target interaction data:
  - raw_data/: Contains raw DTI data from various databases.
  - processed_data/: Contains processed DTI data ready for analysis.

- To preprocess the data:
  ```bash
  python scripts/Preprocessing.py

- To build the DTI pipeline:
  ```bash
  python scripts/DTI_Pipeline.py

- To run predictive modeling and ranking:
  ```bash
  python scripts/Predictive_Modeling_&_Ranking.py

## Citation

If you use this code or associated research in your work, please cite:

Alidoost, Mohammadali and Wilson, L. Jennifer, "Evaluating the Contribution of Drug-Binding Targets to Side Effect Prediction in Drug Development", Submitted (2024).
