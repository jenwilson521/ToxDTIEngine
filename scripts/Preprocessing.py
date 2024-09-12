
####################################################################################################

# UpSet plot

# Prepare the data structure needed for the UpSet plot

import os
import pandas as pd

main_dir = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/'
os.chdir(os.path.join(main_dir, 'Data2/DTIs/DTIs_filtered/'))

file_paths = {
    'Old_DrugBank': 'DTI_OldDrugBank.csv',
    'New_DrugBank': 'DTI_NewDrugBank.csv',
    'ChEMBL': 'DTI_ChEMBL.csv',
    'PubChem': 'DTI_PubChem.csv',
    'STITCH': 'DTI_STITCH.csv',
    'TTD': 'DTI_TTD.csv',
    'Pocket_Features': 'DTI_PocketFeatures.csv'
}

old_drugbank_drugs = [drug.lower() for drug in old_drugbank_baseline]

drug_data = {'Old_DrugBank': old_drugbank_drugs}

for db_name, file_path in file_paths.items():
    if db_name != 'Old_DrugBank': 
        db_df = pd.read_csv(file_path)
        db_drugs_set = set(db_df['Drug'])
        drug_data[db_name] = [1 if drug in db_drugs_set else 0 for drug in old_drugbank_drugs]

comparison_df1 = pd.DataFrame(drug_data)
print(comparison_df1)

# Data should be in a format that indicates the presence (1) or absence (0)

from upsetplot import UpSet
import pandas as pd
import matplotlib.pyplot as plt

comparison_df = comparison_df1

df = pd.DataFrame(comparison_df).set_index('Old_DrugBank')
data_series = df.groupby(list(df.columns)).size()

upset = UpSet(data_series)

axes_dict = upset.plot()
axes_dict['intersections'].set_ylim(0, 175)
plt.show()

####################################################################################################

# Shared drug names

import os
import pandas as pd

main_dir = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/'
os.chdir(os.path.join(main_dir, 'Data2/DTIs/DTIs_filtered/'))

files = {
    'New_DrugBank': 'DTI_NewDrugBank.csv',
    'ChEMBL': 'DTI_ChEMBL.csv',
    'PubChem': 'DTI_PubChem.csv',
    'STITCH': 'DTI_STITCH.csv',
    'TTD': 'DTI_TTD.csv',
    'Pocket_Features': 'DTI_PocketFeatures.csv'
}

drug_sets = {}
for key, filename in files.items():
    df = pd.read_csv(filename)
    df['Drug'] = df['Drug'].str.lower()
    drug_sets[key] = set(df['Drug'])
shared_drugs = set.intersection(*drug_sets.values())
print(shared_drugs)

####################################################################################################

# Heatmap of databases-databases based on the shared drugs

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

main_dir = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/'
os.chdir(os.path.join(main_dir, 'Data2/DTIs/DTIs_filtered/'))

files = {
    'New_DrugBank': 'DTI_NewDrugBank.csv',
    'ChEMBL': 'DTI_ChEMBL.csv',
    'PubChem': 'DTI_PubChem.csv',
    'STITCH': 'DTI_STITCH.csv',
    'TTD': 'DTI_TTD.csv',
    'Pocket_Features': 'DTI_PocketFeatures.csv'
}

drug_sets = {}
for key, filename in files.items():
    df = pd.read_csv(filename)
    drug_sets[key] = set(df['Drug'])

shared_drugs = pd.DataFrame(index=files.keys(), columns=files.keys())
for db1 in files.keys():
    for db2 in files.keys():
        shared_drugs.loc[db1, db2] = len(drug_sets[db1].intersection(drug_sets[db2]))

shared_drugs = shared_drugs.fillna(0).astype(int)

plt.figure(figsize=(10, 8))
sns.heatmap(shared_drugs, annot=True, cmap='cividis', fmt="d")
plt.title('Shared Drugs Among Different Databases')
plt.show()

####################################################################################################

# Heatmap of only shared drugs based on the shared side effects

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

main_dir = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/'
os.chdir(os.path.join(main_dir, 'Data2/DTIs/DTIs_filtered/'))

files = {
    'New_DrugBank': 'DTI_NewDrugBank.csv',
    'ChEMBL': 'DTI_ChEMBL.csv',
    'PubChem': 'DTI_PubChem.csv',
    'STITCH': 'DTI_STITCH.csv',
    'TTD': 'DTI_TTD.csv',
    'Pocket_Features': 'DTI_PocketFeatures.csv'
}

drug_sets = {}
for key, filename in files.items():
    df = pd.read_csv(filename)
    df['Drug'] = df['Drug'].str.lower()
    drug_sets[key] = set(df['Drug'])

shared_drugs = set.intersection(*drug_sets.values())

os.chdir(main_dir)
toxicity_df = pd.read_csv('Drugs_labeled_for_AEs.txt', sep='\t', low_memory=False)
toxicity_df = toxicity_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

side_effects = {col.lower(): [] for col in toxicity_df.columns}
binary_matrix = pd.DataFrame(0, index=list(shared_drugs), columns=side_effects)

for drug in shared_drugs:
    for column in toxicity_df.columns:
        if drug in toxicity_df[column].values:
            binary_matrix.loc[drug, column.lower()] = 1

jaccard_similarity = 1 - squareform(pdist(binary_matrix, 'jaccard'))
similarity_df = pd.DataFrame(jaccard_similarity, index=binary_matrix.index, columns=binary_matrix.index)
sns.clustermap(similarity_df, cmap="cividis", figsize=(12, 12), cbar_kws={'label': 'Similarity Score'})
plt.title('Clustermap of Shared Drugs Based on Side Effects')
output_path = os.path.join(main_dir, 'Data2/DTIs/Drug_info/', 'Clustermap of Shared Drugs Based on Side Effects.png')
plt.savefig(output_path)
plt.show()

####################################################################################################

# Heatmap of only shared drugs based on the shared targets

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

main_dir = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/'
data_dir = os.path.join(main_dir, 'Data2/DTIs/DTIs_filtered/')
os.chdir(data_dir)

files = {
    'New_DrugBank': 'DTI_NewDrugBank.csv',
    'ChEMBL': 'DTI_ChEMBL.csv',
    'PubChem': 'DTI_PubChem.csv',
    'STITCH': 'DTI_STITCH.csv',
    'TTD': 'DTI_TTD.csv',
    'Pocket_Features': 'DTI_PocketFeatures.csv'
}

dbs = []
for db_name, file_name in files.items():
    df = pd.read_csv(file_name)
    df['Database'] = db_name
    dbs.append(df)

all_drugs = pd.concat(dbs)
shared_drugs = all_drugs.groupby('Drug').filter(lambda x: x['Database'].nunique() == len(files))

binary_matrix = pd.get_dummies(shared_drugs.set_index('Drug')['Targets']).groupby(level=0).sum()

jaccard_similarity = 1 - squareform(pdist(binary_matrix, 'jaccard'))
similarity_df = pd.DataFrame(jaccard_similarity, index=binary_matrix.index, columns=binary_matrix.index)
sns.clustermap(similarity_df, cmap="cividis", figsize=(12, 12), cbar_kws={'label': 'Similarity Score'})
plt.title('Clustermap of Shared Drugs Based on targets')
output_path = os.path.join(main_dir, 'Data2/DTIs/Drug_info/', 'Clustermap of Shared Drugs Based on targets.png')
plt.savefig(output_path)
plt.show()

####################################################################################################

# Plot sorted DTIs scatter log

def plot_drug_target_interactions_scatter_log(files):
    """
    Reads the given CSV files, preprocesses data to normalize target lists,
    and generates scatter plots to analyze both the number of targets per drug
    and the number of drugs per target for each database.
    Data is sorted from highest number to lowest, and plotted on a log scale.

    Parameters:
    - files: List of file paths to the CSV files.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    main_dir = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/'
    data_dir = 'Data2/DTIs/DTIs_filtered/'
    output_dir = 'Data2/DTIs/DTIs_comparisons/Drugs&Targets scatter plots/'

    for file in files:
        full_path = os.path.join(main_dir, data_dir, file)
        db = pd.read_csv(full_path)
        db_name = file.split('_')[-1].split('.')[0]

        db['Targets'] = db['Targets'].str.strip("[]").str.split(',')
        db = db.explode('Targets').reset_index(drop=True)
        db['Targets'] = db['Targets'].str.strip(" '\"")
        db['Drug'] = db['Drug'].str.lower()

        targets_per_drug = db.groupby('Drug')['Targets'].nunique().reset_index(name='Target Count').sort_values(by='Target Count', ascending=False)
        drugs_per_target = db.groupby('Targets')['Drug'].nunique().reset_index(name='Drug Count').sort_values(by='Drug Count', ascending=False)

        top_drugs = targets_per_drug.head(10)['Drug'].tolist()
        top_targets = drugs_per_target.head(10)['Targets'].tolist()

        print(f'Top 10 Drugs with most Targets in {db_name}: {top_drugs}')
        print(f'Top 10 Targets with most Drugs in {db_name}: {top_targets}')
        print('')

        targets_per_drug['Drug Index'] = range(1, len(targets_per_drug) + 1)
        drugs_per_target['Target Index'] = range(1, len(drugs_per_target) + 1)

        # Plot targets per drug in log scale
        plt.figure(figsize=(6, 3))
        plt.scatter(targets_per_drug['Drug Index'], targets_per_drug['Target Count'], alpha=0.5, color='blue')
        plt.title(f'{db_name} - Number of Targets per Drug')
        plt.xlabel('Drug Index')
        plt.ylabel('Number of Targets (Log Scale)')
        plt.yscale('log')
        plt.xticks([])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(main_dir, output_dir, f"{db_name}_targets_per_drug_scatter_log.png"))
        plt.show()

        # Plot drugs per target in log scale
        plt.figure(figsize=(6, 3))
        plt.scatter(drugs_per_target['Target Index'], drugs_per_target['Drug Count'], alpha=0.5, color='red')
        plt.title(f'{db_name} - Number of Drugs per Target')
        plt.xlabel('Target Index')
        plt.ylabel('Number of Drugs (Log Scale)')
        plt.yscale('log')
        plt.xticks([])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(main_dir, output_dir, f"{db_name}_drugs_per_target_scatter_log.png"))
        plt.show()

    return

####################################################################################################
