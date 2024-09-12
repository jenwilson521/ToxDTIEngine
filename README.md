ToxDTIEngine
Overview
ToxDTIEngine is a Python-based tool for predicting drug-induced side effects using Drug-Target Interaction (DTI) data. The project integrates raw and processed data from multiple databases, builds a DTI pipeline, and uses predictive modeling to rank and evaluate side effect risks for drug targets.

Data Structure
data/raw_data/: Contains raw drug-target interaction data sourced from various databases.
data/processed_data/: Contains processed drug-target interaction data.
scripts/: Python scripts used for preprocessing, DTI pipeline building, and predictive modeling.
Prerequisites
Ensure you have the following dependencies installed:

Python 3.x
Required packages:
bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn
Usage
Preprocessing Data: The first step is to clean and prepare raw data for use.

bash
Copy code
python scripts/Preprocessing.py
Building the DTI Pipeline: Next, integrate Drug-Target Interaction data across multiple databases.

bash
Copy code
python scripts/DTI_Pipeline.py
Predictive Modeling and Ranking: Finally, use machine learning models to predict drug side effects and rank the targets.

bash
Copy code
python scripts/Predictive_Modeling_&_Ranking.py
Folder Structure
kotlin
Copy code
ToxDTIEngine/
├── data/
│   ├── raw_data/
│   └── processed_data/
├── scripts/
│   ├── Preprocessing.py
│   ├── DTI_Pipeline.py
│   └── Predictive_Modeling_&_Ranking.py
└── README.md
Citation
If you use this code or associated research, please cite:

Alidoost, Mohammadali and Wilson, L. Jennifer, "Preclinical Side Effect Prediction through Pathway Engineering of Protein Interaction Network Models", Submitted (2023).
