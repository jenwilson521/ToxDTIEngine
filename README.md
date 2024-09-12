# ToxDTIEngine

## Overview

This repository, **ToxDTIEngine**, contains the code and associated resources for the research presented in the manuscript titled "**Evaluating the Contribution of Drug-Binding Targets to Side Effect Prediction in Drug Development**". The study focuses on predicting drug-induced side effects using protein-protein interactions (PPIs) by incorporating drug-target interaction (DTI) data from multiple databases. **ToxDTIEngine** uses another tool called [**PathFX**](https://github.com/jenwilson521/PathFX), which is a PPI tool, and builds a DTI pipeline employing predictive modeling to select drug targets and rank databases to predict side effects.

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
- For setup guidance, please visit [**PathFX**](https://github.com/jenwilson521/PathFX) and follow the provided instructions.

## Usage

Detailed instructions on how to use the code:

- Clone this repository to your local machine or cluster:
  ```bash
  git clone https://github.com/yourusername/ToxDTIEngine.git

- To be able to run the last version of PathFX made in our analyses, first, you need to clone [**PathFX**](https://github.com/jenwilson521/PathFX). Afterward, you should add (copy/paste) the files available in the 'pathfx/' folder here in this GitHub repository to the same folder names (data/scripts/rscs/results) in your cloned PathFX folder on your local drive. Subsequently, you can use the 'runpathfx_scr.py' script in our 'scripts/' folder to run the last version of PathFX on your operating system and re-generate the results.

- The data/ folder contains both raw and processed drug-target interaction data:
  - raw_data/: Contains raw DTI data from various databases.
    - The 'DrugToxicity_data.txt' file in the 'data/raw_data/' folder is the dataset we used for our analysis, consisting of pairs of drugs and their associated side effects obtained from drug labels. 
  - processed_data/: Contains processed DTI data ready for analysis.

- To preprocess the data:
  ```bash
  python scripts/Preprocessing.py

- To build the DTI pipeline and to evaluate:
  ```bash
  python scripts/DTI_Pipeline_Evaluation.py

- To run predictive modeling and to select targets/rank databases:
  ```bash
  python scripts/Predictive_Modeling_Ranking.py

- Make sure to update the directory paths in all scripts to match your local environment before running them.

## Citation

If you use this code or associated research in your work, please cite:

Alidoost, Mohammadali and Wilson, L. Jennifer, "Evaluating the Contribution of Drug-Binding Targets to Side Effect Prediction in Drug Development", Submitted (2024).
