# ToxDTIEngine

## Overview

This repository, **ToxDTIEngine**, contains the code and associated resources for the research presented in the manuscript "**Evaluating the Contribution of Drug-Binding Targets to Side Effect Prediction in Drug Development**". The study focuses on predicting drug-induced side effects using protein-protein interactions (PPIs) by incorporating drug-target interaction (DTI) data from multiple databases. **ToxDTIEngine** uses another tool called [PathFX](https://github.com/jenwilson521/PathFX), which is a PPI tool, and builds a DTI pipeline employing predictive modeling to select drug targets and rank databases to predict side effects.

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
- For setup guidance, please visit [PathFX](https://github.com/jenwilson521/PathFX) and follow the provided instructions.

## Usage

Detailed instructions on how to use the code:

- Clone this repository to your local machine or cluster:
  ```bash
  git clone https://github.com/yourusername/ToxDTIEngine.git

- To be able to run the last version of PathFX made in our analyses, first, you need to clone [PathFX](https://github.com/jenwilson521/PathFX). Afterward, you should add (copy/paste) the files available in the 'pathfx/' folder here in this GitHub repository to the same folder names (data/scripts/rscs/results) in your cloned PathFX folder on your local drive.
- To run the last version of PathFX on your operating system and re-generate the results:
  ```bash
  python scripts/RunPathFX.py
  
- The data/ folder contains all the datasets used in our analysis, organized as follows:

    - DrugBank: We utilized DrugBank (Release Version 5.1.10), referred to as "New_DrugBank" in our study. Various versions of DrugBank can be accessed here: [DrugBank Releases](https://go.drugbank.com/releases). To parse DrugBank data, we referred to the DrugBank Parsing Guide notebook: [DrugBank Parsing Guide](https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb). We adapted and used the following code to first extract the information (data/parse.ipynb) and later used (data/filter) to create 'DTI_NewDrugBank.csv', which is then used with "/scripts/Preprocessing.py". Importantly, this site requires registration to download the drugbank.xml.gz file.
 
    - ChEMBL: We extracted drug candidates and their proposed therapeutic targets from the drug mechanisms table. We specifically downloaded "chembl_33.db" from [ChEMBLdb-latest](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/). More details can be found via the interface here: [ChEMBL Mechanisms of Action](https://www.ebi.ac.uk/chembl/g/#browse/mechanisms_of_action). All targets are linked to unique ChEMBL IDs ("raw_data/target_ChEMBL_ID.txt"), and we used UniProt accessions to identify protein targets. SQL queries for extracting this data are provided in the code_chembl.py file, in the "raw_data/" folder.
 
    - PubChem: We retrieved small-molecule drugs from the PubChem compound database, focusing on FDA Orange Book records. You can access this information here: [FDA Orange Book in PubChem](https://pubchem.ncbi.nlm.nih.gov/classification/#hid=72) (Expand the "Information Sources" submenu and click the link next to the FDA Orange Book). We specifically identified CIDs ("raw_data/cid_list_pubchem.txt") that overlapped with the DTD and our baseline analysis and generated a file, "cids.txt", which is included in "data/raw_data/". To download compound-target interaction data, use the script "download_pubchem.sh" provided in the "raw_data/" folder.
 
    - STITCH: The chemicals (chemicals.v5.0.tsv), chemical-protein links (9606.protein_chemical.links.v5.0.tsv), and corresponding STRING proteins (9606.protein.info.v12.0.txt) were downloaded from these sources: [STITCH Download](http://stitch.embl.de/cgi/download.pl?UserId=M1MWuGzm9DP9&sessionId=BuMJ6vly8bv4&species_text=Homo+sapiens) & [STRING Download](https://string-db.org/cgi/download?sessionId=b4gDpE1CkXQH&species_text=Homo+sapiens). A script for processing these large datasets, code_stitch_map.py, is available in the "raw_data/" folder.
 
    - Therapeutic Target Database (TTD): We downloaded drug IDs (i.e., “P1-03-TTD_crossmatching.txt”) and DTI (i.e., “P1-07-Drug-TargetMapping.xlsx”) data for “successful” (i.e., P2-02-TTD_uniprot_successful.txt) and “clinical” (i.e., “P2-03-TTD_uniprot_clinical.txt”) from the [Therapeutic Target Database](https://db.idrblab.net/ttd/full-data-download). We have provided our code for parsing these files in "data/raw_data/".
 
    - Pocket Features: Predicted drug-binding interactions were derived from Liu & Altman, 2011. You can review the study here: [Liu & Altman (2011)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002326).
 
    - Pocket Features: Predicted drug-binding interactions were derived from Liu & Altman, 2011. More details about the algorithm are provided here: [Liu & Altman (2011)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002326). Although the data was published with Wilson et al. 2018, we have provided an additional copy here (/data/raw_data/drug_gene_int_scores_Pocket Features.txt) and also provided code for parsing the data in the "raw_data/" folder.



  - processed_data/:
    - This folder contains sample processed DTI data ready for analysis. You can use this example as a template for generating processed files from each database before combining them for the analysis pipeline.
      
  - DrugToxicity_data.txt:
    - This file contains the dataset we used in our analysis, consisting of drug-side effect pairs sourced from drug labels. The original data can be found at [Designated Medical Event Pathways GitHub](https://github.com/jenwilson521/Designated-Medical-Event-Pathways), published in Wilson et al., [CPT: Pharmacometrics & Systems Pharmacology, 2022](https://ascpt.onlinelibrary.wiley.com/doi/10.1002/psp4.12861).

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
