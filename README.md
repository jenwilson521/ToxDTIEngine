# ToxDTIEngine

## Overview

This repository, **ToxDTIEngine**, contains the code and associated resources for the research presented in the manuscript "**Evaluating the Contribution of Drug-Binding Targets to Side Effect Prediction in Drug Development**". The study focuses on predicting drug-induced side effects using protein-protein interactions (PPIs) by incorporating drug-target interaction (DTI) data from multiple databases. **ToxDTIEngine** uses another tool called [**PathFX**](https://github.com/jenwilson521/PathFX), which is a PPI tool, and builds a DTI pipeline employing predictive modeling to select drug targets and rank databases to predict side effects.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Citation](#citation)

## Getting Started

These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes. See [Usage](#usage) for notes on using the code in a research context.

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

- To be able to run the last version of PathFX made in our analyses, first, you need to clone [**PathFX**](https://github.com/jenwilson521/PathFX). Afterward, you should add (copy/paste) the files available in the 'pathfx/' folder here in this GitHub repository to the same folder names (data/scripts/rscs/results) in your cloned PathFX folder on your local drive.
- To run the last version of PathFX on your operating system and re-generate the results:
  ```bash
  python scripts/RunPathFX.py
  
- The data/ folder provides information about the data used in our analysis:
  - raw_data/:
    - DrugBank: We used DrugBank (Release Version 5.1.10), referred to as “New_DrugBank” throughout our study. You can download different versions of the DrugBank data here: https://go.drugbank.com/releases. You can parse the DrugBank data using this notebook as a guide: https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb.
    - ChEMBL: We used the drugs/clinical candidates with their proposed therapeutic targets from the data stored in the drug mechanisms table. You can find the information on the interface here: https://www.ebi.ac.uk/chembl/g/#browse/mechanisms_of_action. ChEMBL targets were all associated with a unique target ChEMBL_ID. We also used UniProt accessions as our primary identifier for protein targets. Refer to the "code_chembl.py" file in the raw_data/ folder where we provided the SQL queries to extract and filter the complementary data.
    - PubChem: We downloaded the small molecule drugs in the PubChem compound database, the FDA Orange Book records: https://pubchem.ncbi.nlm.nih.gov/classification/#hid=72 (open up the “Information Sources” submenu, then click the count next to the FDA Orange Book). Afterward, you can loop over your list of drugs (with PubChem CID). You can run the file in the raw_data/ folder, "download_pubchem.sh", to download the compound-target interaction data.
    - STITCH: We downloaded the chemical-protein links and list of STRING proteins from these links: http://stitch.embl.de/cgi/download.pl?UserId=M1MWuGzm9DP9&sessionId=BuMJ6vly8bv4&species_text=Homo+sapiens & https://string-db.org/cgi/download?sessionId=b4gDpE1CkXQH&species_text=Homo+sapiens. To process the large data files, we developed a code that you can find in the raw_data/ folder, "code_stitch_map.py".
    - Therapeutic_Target_Database (TTD): We downloaded the drug, target, and DTI information from this link: https://db.idrblab.net/ttd/full-data-download.
    - Pocket_features: We used the predicted drug-binding interactions generated from Liu & Altman, 2011, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002326.
  - processed_data/: Contains dummy processed DTI data, as an example that is ready for analysis. You should make such a file for each database and combine them all to run the pipeline.
  - The 'DrugToxicity_data.txt' file in the 'data/' folder is the dataset we used for our analysis. It consists of pairs of drugs and their associated side effects obtained from drug labels. 

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
