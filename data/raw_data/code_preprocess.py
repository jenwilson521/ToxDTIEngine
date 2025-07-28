####################################################################################################

# This full script describes the preprocessing steps of the six drug-target interaction databases.
# For each database:
    # inputs: various drug and target files 
    # output: the filtered DTI

####################################################################################################

##### Database: New_DrugBank

import os
import pandas as pd

main_dir = '...'
os.chdir(os.path.join(main_dir, 'Data2/DrugBank/'))

# proteins.tsv

file_path = 'proteins.tsv'
df_db_proteins = pd.read_csv(file_path, delimiter='\t')
print('shape:', df_db_proteins.shape)
print('columns:', df_db_proteins.columns)
print('')

# drugbank.tsv

file_path = 'drugbank.tsv'
df_db_drugbank = pd.read_csv(file_path, delimiter='\t')
print('shape:', df_db_drugbank.shape)
print('columns:', df_db_drugbank.columns)
print('')

# drug_target_identifiers_all.csv

file_path = 'drug_target_identifiers_all.csv'
df_db_drugtarget = pd.read_csv(file_path)
print('shape:', df_db_drugtarget.shape)
print('columns:', df_db_drugtarget.columns)
print('')

# merge

merged_df = pd.merge(df_db_proteins, df_db_drugtarget, left_on='uniprot_id', right_on='UniProt ID', how='inner')
merged_df = pd.merge(merged_df, df_db_drugbank[['drugbank_id', 'name']], on='drugbank_id', how='left')
new_df = merged_df[['name', 'Gene Name']]
print('shape:', new_df.shape)
print('columns:', new_df.columns)
print('')

# unique drugs, their targets, and target counts

unique_drugs_count = new_df['name'].nunique()
df_chemical_targets = new_df.groupby('name')['Gene Name'].agg(['count', 'unique']).reset_index()
df_chemical_targets.columns = ['Drug', 'TargetCount', 'UniqueTargets']
print('# of unique drugs:', df_chemical_targets['Drug'].nunique())
print('')

# filter drugs with more than one target

drugs_with_multiple_targets = df_chemical_targets[df_chemical_targets['TargetCount'] > 1]
print('# of unique drugs with more than one target:', drugs_with_multiple_targets['Drug'].nunique())
print('')
target_counts = drugs_with_multiple_targets['TargetCount'].tolist()
if target_counts:
    max_target_count = max(target_counts)
    print('max # of targets for a drug:', max_target_count)
else:
    print('no drugs with multiple targets found.')
print('')

# shared drugs with the baseline analysis

drug_stitch_ls = df_chemical_targets['Drug'].tolist()
drug_stitch_filtered = [str(x).lower() for x in drug_stitch_ls if isinstance(x, str)]
drug_stitch_set = set(drug_stitch_filtered)
drug_all_filtered = [str(x).lower() for x in drug_all_ls if isinstance(x, str)]
drug_all_set = set(drug_all_filtered)
shared_drugs = drug_stitch_set.intersection(drug_all_set)
print('# of shared drugs:', len(shared_drugs))
print('shared drugs between the DrugBank drugs & the baseline analysis:', shared_drugs)
print('')

# unique targets

df_chemical_targets
max_target_row = df_chemical_targets[df_chemical_targets['TargetCount'] == 328]
UniqueTargets_ls = max_target_row['UniqueTargets'].tolist()
print('UniqueTargets_ls:', len(UniqueTargets_ls[0]))
print('')

# 'organism' = 'Humans'

filtered_df = df_db_proteins[df_db_proteins['organism'] == 'Humans'].copy()

# merge

merged_df = pd.merge(filtered_df, df_db_drugtarget, left_on='uniprot_id', right_on='UniProt ID', how='inner')
merged_df = pd.merge(merged_df, df_db_drugbank[['drugbank_id', 'name']], on='drugbank_id', how='left')
new_df = merged_df[['name', 'Gene Name']]
print('shape:', new_df.shape)
print('columns:', new_df.columns)
print('')

# unique drugs, their targets, and target counts

unique_drugs_count = new_df['name'].nunique()
df_chemical_targets = new_df.groupby('name')['Gene Name'].agg(['count', 'unique']).reset_index()
df_chemical_targets.columns = ['Drug', 'TargetCount', 'UniqueTargets']
print('# of unique drugs:', df_chemical_targets['Drug'].nunique())
print('')

# filter drugs with more than one target

drugs_with_multiple_targets = df_chemical_targets[df_chemical_targets['TargetCount'] > 1]
print('# of unique drugs with more than one target:', drugs_with_multiple_targets['Drug'].nunique())
print('')
target_counts = drugs_with_multiple_targets['TargetCount'].tolist()
if target_counts:
    max_target_count = max(target_counts)
    print('max # of targets for a drug:', max_target_count)
else:
    print('no drugs with multiple targets found.')
print('')

# shared drugs with the baseline analysis

drug_stitch_ls = df_chemical_targets['Drug'].tolist()
drug_stitch_filtered = [str(x).lower() for x in drug_stitch_ls if isinstance(x, str)]
drug_stitch_set = set(drug_stitch_filtered)
drug_all_filtered = [str(x).lower() for x in drug_all_ls if isinstance(x, str)]
drug_all_set = set(drug_all_filtered)
shared_drugs = drug_stitch_set.intersection(drug_all_set)
print('# of shared drugs:', len(shared_drugs))
print('shared drugs between the DrugBank drugs & the baseline analysis:', shared_drugs)
print('')

# unique targets

df_chemical_targets
max_target_row = df_chemical_targets[df_chemical_targets['TargetCount'] == 326]
UniqueTargets_ls = max_target_row['UniqueTargets'].tolist()
print('UniqueTargets_ls:', len(UniqueTargets_ls[0]))
print('')

# saving the DTIs to a file in a new directory

df = new_df.copy()
shared_drugs = shared_drugs

df = df.rename(columns={'name': 'Drug', 'Gene Name': 'Targets'})
df['Drug'] = df['Drug'].str.lower()
filtered_df = df.loc[df['Drug'].isin(shared_drugs)].copy()

grouped_data = (
    filtered_df.groupby('Drug')['Targets']
    .apply(list)
    .reset_index()
)

output_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/Data2/DTIs/DTI_NewDrugBank.csv'
grouped_data.to_csv(output_path, index=False)

####################################################################################################

##### Database: ChEMBL

import os
import pandas as pd
import ast

main_dir = '...'
os.chdir(os.path.join(main_dir, 'Data2/Chembl/'))

# the drug mechanisms table

file_path = 'chembl_mechanisms_of_action.csv'
df_chembl_moa = pd.read_csv(file_path, delimiter=';')
print('chembl_mechanisms_of_action:')
print('')
print('shape:', df_chembl_moa.shape)
print('columns:', df_chembl_moa.columns)
df_chembl_drug_target = df_chembl_moa[['Parent Molecule Name', 'Target Name']]
print('')

# unique drugs, their targets, and target counts

unique_drugs_count = df_chembl_drug_target['Parent Molecule Name'].nunique()
df_chembl_drug_targets = df_chembl_drug_target.groupby('Parent Molecule Name')['Target Name'].agg(['count', 'unique']).reset_index()
df_chembl_drug_targets.columns = ['Drug', 'TargetCount', 'UniqueTargets']
print('# of unique drugs:', df_chembl_drug_targets['Drug'].nunique())
print('')

# filter drugs with more than one target

drugs_with_multiple_targets = df_chembl_drug_targets[df_chembl_drug_targets['TargetCount'] > 1]
print('# of unique drugs with more than one target:', drugs_with_multiple_targets['Drug'].nunique())
print('')
target_counts = drugs_with_multiple_targets['TargetCount'].tolist()
if target_counts:
    max_target_count = max(target_counts)
    print('max # of targets for a drug:', max_target_count)
else:
    print('no drugs with multiple targets found.')
print('')

# shared drugs between chembl & the baseline analysis

drug_chembl_ls = df_chembl_drug_targets['Drug'].tolist()
drug_chembl_filtered = [str(x).lower() for x in drug_chembl_ls if isinstance(x, str)]
drug_chembl_set = set(drug_chembl_filtered)
drug_all_filtered = [str(x).lower() for x in drug_all_ls if isinstance(x, str)]
drug_all_set = set(drug_all_filtered)
shared_drugs = drug_chembl_set.intersection(drug_all_set)
print('# of shared drugs:', len(shared_drugs))
print('shared drugs between chembl drugs & the baseline analysis:', shared_drugs)
shared_filtered_df = drug_chembl_set.intersection(shared_drugs_humans)
print('shared drugs between chembl drugs & the human baseline analysis:',len(filtered_df))
print('')

# unique targets

df_chembl_drug_targets
max_target_row = df_chembl_drug_targets[df_chembl_drug_targets['TargetCount'] == 15]
UniqueTargets_ls = max_target_row['UniqueTargets'].tolist()
print('UniqueTargets_ls:', len(UniqueTargets_ls[0]))
print('')

# saving the DTIs to a file in a new directory

df = df_chembl_drug_target.copy()
shared_drugs = shared_filtered_df
df = df.rename(columns={'Parent Molecule Name': 'Drug', 'Target Name': 'Targets'})
df['Drug'] = df['Drug'].str.lower()
filtered_df = df.loc[df['Drug'].isin(shared_drugs)].copy()

grouped_data = (
    filtered_df.groupby('Drug')['Targets']
    .apply(list)
    .reset_index()
)
output_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/Data2/DTIs/dti_chembl.csv'
grouped_data.to_csv(output_path, index=False)

# find the short-form target names

file_path = 'chembl_mechanisms_of_action.csv'
df_chembl_moa = pd.read_csv(file_path, delimiter=';')
df_chembl_drug_target2 = df_chembl_moa[['Parent Molecule Name', 'Target ChEMBL ID', 'Target Name']]
df = df_chembl_drug_target2.copy()
shared_drugs = shared_filtered_df
df = df.rename(columns={'Parent Molecule Name': 'Drug', 'Target ChEMBL ID': 'Target ChEMBL ID', 'Target Name': 'Targets'})
df['Drug'] = df['Drug'].str.lower()
filtered_df = df.loc[df['Drug'].isin(shared_drugs)].copy()

target_ChEMBL_ID_column = df['Target ChEMBL ID'].dropna().tolist()

with open('target_ChEMBL_ID.txt', 'w') as file:
    for id in target_ChEMBL_ID_column:
        file.write(f"{id}\n")

file_path = 'chembl_output.csv'
df_chembl_output = pd.read_csv(file_path)
df_chembl_targets = df_chembl_output[['target_chembl_ID', 'component_synonym']]

merged_df = pd.merge(filtered_df, df_chembl_targets, left_on='Target ChEMBL ID', right_on='target_chembl_ID', how='left')
final_df = merged_df[['Drug', 'component_synonym']].rename(columns={'component_synonym': 'Targets'})

grouped_data = (
    final_df.groupby('Drug')['Targets']
    .apply(lambda x: list(set(x.dropna())))  # Remove NaNs and collect unique targets using set
    .reset_index()
)
grouped_data['Targets'] = grouped_data['Targets'].apply(lambda x: str([str(target) for target in x]))

# analyze ChEMBL

df = grouped_data
df['Targets'] = df['Targets'].apply(lambda x: list(set(x)))
df['TargetCount'] = df['Targets'].apply(len)
new_df = df[['Drug', 'Targets', 'TargetCount']]
new_df_sorted = new_df.sort_values(by='TargetCount', ascending=False)
new_df_sorted['Targets'] = new_df_sorted['Targets'].apply(lambda x: str([str(target) for target in x]))
ChEMBL_data = new_df_sorted[new_df_sorted['TargetCount'] > 0]
top_quartile_threshold_ChEMBL = ChEMBL_data['TargetCount'].quantile(0.98)
ChEMBL_filtered_data = ChEMBL_data[ChEMBL_data['TargetCount'] <= top_quartile_threshold_ChEMBL]
ChEMBL_filtered_data.drop('TargetCount', axis=1, inplace=True)
output_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/Data2/DTIs/DTIs_filtered/DTI_ChEMBL.csv'
ChEMBL_filtered_data.to_csv(output_path, index=False)

####################################################################################################

##### Database: PubChem

import os
import pandas as pd

main_dir = '...'
os.chdir(os.path.join(main_dir, 'Data2/PubChem/'))

# Chemical-Target Interactions for only Aspirin

file_path = 'pubchem_compoundtarget_asprin.csv'
df_pubchem = pd.read_csv(file_path)
print('shape:', df_pubchem.shape)
print('columns:', df_pubchem.columns)
df_pubchem_drug_target = df_pubchem[['cmpdname', 'srctargetname']]
print('')

# Compounds

file_path = 'PubChem_compound_list.csv'
df_pubchem = pd.read_csv(file_path)
print('shape:', df_pubchem.shape)
print('columns:', df_pubchem.columns)
df_pubchem_drug_target = df_pubchem[['cid', 'cmpdname']]
print('')

df_pubchem_drug_target['cmpdname'] = df_pubchem_drug_target['cmpdname'].str.lower()
filtered_df = df_pubchem_drug_target[df_pubchem_drug_target['cmpdname'].isin(shared_drugs_humans)]
cid_list = filtered_df['cid'].tolist()
with open("cid_list.txt", "w") as f:
    for cid in cid_list:
        f.write(str(cid) + "\n")

df = df_pubchem_drug_target
unique_cmpdname_count = df['cmpdname'].nunique()

cmpdname_gene_count = {}
for index, row in df.iterrows():
    cmpdname = row['cmpdname']
    genes = row['genename']

    gene_count = len(genes)

    if cmpdname not in cmpdname_gene_count:
        cmpdname_gene_count[cmpdname] = 0
    cmpdname_gene_count[cmpdname] += gene_count

print(f"Number of unique cmpdname: {unique_cmpdname_count}")
print(f"Number of genes for each cmpdname: {cmpdname_gene_count}")
print('')

drugs_with_more_than_1_target = df[df['genename'].apply(len) > 1]['cmpdname'].nunique()
max_targets_per_drug = df['genename'].apply(len).max()

print(f"Number of unique drugs with more than 1 target: {drugs_with_more_than_1_target}")
print(f"Maximum number of unique targets for a drug: {max_targets_per_drug}")
print('')

# saving the DTIs to a file in a new directory

df = df_pubchem_drug_target.copy()
df = df.rename(columns={'cmpdname': 'Drug', 'genename': 'Targets'})
df['Drug'] = df['Drug'].str.lower()
filtered_df = df.loc[df['Drug'].isin(shared_drugs)].copy()

grouped_data = (
    filtered_df.groupby('Drug')['Targets']
    .apply(list)
    .reset_index()
)

grouped_data = df
output_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/Data2/DTIs/dti_pubchem.csv'
grouped_data.to_csv(output_path, index=False)

####################################################################################################

##### Database: STITCH

import os
import pandas as pd

main_dir = '...'
os.chdir(os.path.join(main_dir, 'Data2/STITCH/'))

# 9606.protein_chemical.links.v5.0.tsv

file_path = '9606.protein_chemical.links.v5.0.tsv'
df_protein_chemical = pd.read_csv(file_path, delimiter='\t')
print('shape:', df_protein_chemical.shape)
print('columns:', df_protein_chemical.columns)
print('')

# 9606.protein_chemical.links.v5.0.tsv

file_path = '9606.protein_chemical.links.v5.0.tsv'
df_protein_chemical = pd.read_csv(file_path, delimiter='\t')
print('shape:', df_protein_chemical.shape)
print('columns:', df_protein_chemical.columns)
print('')

# 9606.protein_chemical.links.v5.0.tsv

file_path = '9606.protein_chemical.links.v5.0.tsv'
df_protein_chemical = pd.read_csv(file_path, delimiter='\t')
print('shape:', df_protein_chemical.shape)
print('columns:', df_protein_chemical.columns)
print('')

# combine the previous 2 dataframes

merged_df = df_protein_chemical.merge(df_protein_info, left_on='protein', right_on='#string_protein_id', how='inner')
stitch_chemicalid_protein = merged_df[['chemical', 'preferred_name']]
stitch_chemicalid_protein = stitch_chemicalid_protein.rename(columns={'preferred_name': 'target'})
#stitch_chemicalid_protein.to_csv('stitch_base.tsv', sep='\t', index=False)

# read a tsv large file:
  # splitted into 20 files:
  # added columns to all other 19 files, split_2-split_20
#shape: (6157348, 4)
#columns: Index(['chemical', 'name', 'molecular_weight', 'SMILES_string'], dtype='object')
#       chemical                               name  molecular_weight                     SMILES_string
#0  CIDs00000001                    acetylcarnitine         203.23558  CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C
#1  CIDs00000003  2,3-dihydro-2,3-dihydroxybenzoate         156.13602          C1=CC(C(C(=C1)C(=O)O)O)O

file_path = 'stitch_chem_targ.tsv'
df_chemical_target = pd.read_csv(file_path, delimiter='\t')
print('shape:', df_chemical_target.shape)
print('columns:', df_chemical_target.columns)
print('')

# unique drugs, their targets, and target counts

unique_drugs_count = df_chemical_target['chemical_name'].nunique()
df_chemical_targets = df_chemical_target.groupby('chemical_name')['target_name'].agg(['count', 'unique']).reset_index()
df_chemical_targets.columns = ['Drug', 'TargetCount', 'UniqueTargets']
print('# of unique drugs:', df_chemical_targets['Drug'].nunique())
print('')

# filter drugs with more than one target

drugs_with_multiple_targets = df_chemical_targets[df_chemical_targets['TargetCount'] > 1]
print('# of unique drugs with more than one target:', drugs_with_multiple_targets['Drug'].nunique())
print('')
target_counts = drugs_with_multiple_targets['TargetCount'].tolist()
if target_counts:
    max_target_count = max(target_counts)
    print('max # of targets for a drug:', max_target_count)
else:
    print('no drugs with multiple targets found.')
print('')

# shared drugs with the baseline analysis

drug_stitch_ls = df_chemical_targets['Drug'].tolist()
drug_stitch_filtered = [str(x).lower() for x in drug_stitch_ls if isinstance(x, str)]
drug_stitch_set = set(drug_stitch_filtered)
drug_all_filtered = [str(x).lower() for x in drug_all_ls if isinstance(x, str)]
drug_all_set = set(drug_all_filtered)
shared_drugs = drug_stitch_set.intersection(drug_all_set)
print('# of shared drugs:', len(shared_drugs))
print('shared drugs between STITCH drugs & the baseline analysis:', shared_drugs)
shared_filtered_df = drug_stitch_set.intersection(shared_drugs_humans)
print('shared drugs between STITCH drugs & the human baseline analysis:',len(shared_filtered_df))
STITCH_Drugs = shared_filtered_df
print(STITCH_Drugs)
print('')

# unique targets

df_chemical_targets
max_target_row = df_chemical_targets[df_chemical_targets['TargetCount'] == 12020]
UniqueTargets_ls = max_target_row['UniqueTargets'].tolist()
print('UniqueTargets_ls:', len(UniqueTargets_ls[0]))
print('')

# saving the DTIs to a file in a new directory

df = df_chemical_target.copy()
shared_drugs = shared_filtered_df
df = df.rename(columns={'chemical_name': 'Drug', 'target_name': 'Targets'})
df['Drug'] = df['Drug'].str.lower()
filtered_df = df.loc[df['Drug'].isin(shared_drugs)].copy()

grouped_data = (
    filtered_df.groupby('Drug')['Targets']
    .apply(list)
    .reset_index()
)

output_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/Data2/DTIs/dti_stitch.csv'
grouped_data.to_csv(output_path, index=False)

####################################################################################################

##### Database: TTD

import os
import pandas as pd
import re

main_dir = '...'
os.chdir(os.path.join(main_dir, 'Data2/TTD/'))

# drug-target ID pairs
# P1-07-Drug-TargetMapping file

file_path = 'P1-07-Drug-TargetMapping.xlsx'
df = pd.read_excel(file_path)
print(df.columns)
print(df.shape)
print('')
df_ttd_ids = df[['DrugID', 'TargetID']]
print('shape of drug-target IDs:', df_ttd_ids.shape)
print('')

# map target IDs
# P2-02-TTD_uniprot_successful

data = {}
file_path = 'P2-02-TTD_uniprot_successful.txt'
with open(file_path, 'r') as file:
    current_key = None
    for line in file:
        line = line.strip()

        if line.startswith('T'):
            current_key = line
            data[current_key] = {}
        else:
            elements = line.split('\t')
            if len(elements) == 2:
                key, value = elements
                data[current_key][key] = value
data_dict = data
target_ids = []
targnames = []
current_target_id = None
for key, value in data_dict.items():
    parts = key.split('\t')
    if parts[1] == 'TARGETID':
        current_target_id = parts[2]
    elif parts[1] == 'TARGNAME':
        target_ids.append(current_target_id)
        targnames.append(parts[2])
df_t1 = pd.DataFrame({'TARGETID': target_ids, 'TARGNAME': targnames})

# P2-03-TTD_uniprot_clinical

data = {}
file_path = 'P2-03-TTD_uniprot_clinical.txt'
with open(file_path, 'r') as file:
    current_key = None
    for line in file:
        line = line.strip()

        if line.startswith('T'):
            current_key = line
            data[current_key] = {}
        else:
            elements = line.split('\t')
            if len(elements) == 2:
                key, value = elements
                data[current_key][key] = value
data_dict = data
target_ids = []
targnames = []
current_target_id = None
for key, value in data_dict.items():
    parts = key.split('\t')
    if parts[1] == 'TARGETID':
        current_target_id = parts[2]
    elif parts[1] == 'TARGNAME':
        target_ids.append(current_target_id)
        targnames.append(parts[2])
df_t2 = pd.DataFrame({'TARGETID': target_ids, 'TARGNAME': targnames})

# successful + clinical

df_t = pd.concat([df_t1, df_t2], ignore_index=True)
print('shape of target IDs:', df_t.shape)
print('')

# map drug IDs
# P1-03-TTD_crossmatching

data = {}
file_path = 'P1-03-TTD_crossmatching.txt'
with open(file_path, 'r') as file:
    current_id = None
    for line in file:
        line = line.strip()
        if line.startswith('D'):
            current_id = line
            data[current_id] = {}
        else:
            parts = line.split('\t', 1)
            if len(parts) == 2:
                key, value = parts
                data[current_id][key] = value
data_dict = data
identifiers = []
drugnames = []
for key, value in data_dict.items():
    parts = key.split('\t')
    if len(parts) == 3 and parts[1] == 'DRUGNAME':
        identifiers.append(parts[0])
        drugnames.append(parts[2])
df_d = pd.DataFrame({'Identifier': identifiers, 'DrugName': drugnames})
print('shape of drug IDs:', df_d.shape)
print('')

# map & merge dataframes

merged_df1 = pd.merge(df_ttd_ids, df_d, left_on='DrugID', right_on='Identifier', how='inner')
merged_df = pd.merge(merged_df1, df_t, left_on='TargetID', right_on='TARGETID', how='inner')
df_ttd_drug_target = merged_df[['DrugName', 'TARGNAME']]

# unique drugs, their targets, and target counts

unique_drugs_count = df_ttd_drug_target['DrugName'].nunique()
df_ttd_drug_targets = df_ttd_drug_target.groupby('DrugName')['TARGNAME'].agg(['count', 'unique']).reset_index()
df_ttd_drug_targets.columns = ['Drug', 'TargetCount', 'UniqueTargets']
print('# of unique drugs:', df_ttd_drug_targets['Drug'].nunique())
print('')

# filter drugs with more than one target

drugs_with_multiple_targets = df_ttd_drug_targets[df_ttd_drug_targets['TargetCount'] > 1]
print('# of unique drugs with more than one target:', drugs_with_multiple_targets['Drug'].nunique())
print('')
target_counts = drugs_with_multiple_targets['TargetCount'].tolist()
if target_counts:
    max_target_count = max(target_counts)
    print('max # of targets for a drug:', max_target_count)
else:
    print('no drugs with multiple targets found.')
print('')

# shared drugs with the baseline analysis

drug_ttd_ls = df_ttd_drug_targets['Drug'].tolist()
drug_ttd_filtered = [str(x).lower() for x in drug_ttd_ls if isinstance(x, str)]
drug_ttd_set = set(drug_ttd_filtered)
drug_all_filtered = [str(x).lower() for x in drug_all_ls if isinstance(x, str)]
drug_all_set = set(drug_all_filtered)
shared_drugs = drug_ttd_set.intersection(drug_all_set)
print('# of shared drugs:', len(shared_drugs))
print('shared drugs between ttd drugs & the baseline analysis:', shared_drugs)
shared_filtered_df = drug_ttd_set.intersection(shared_drugs_humans)
print('shared drugs between TTD drugs & the human baseline analysis:',len(shared_filtered_df))
print(shared_filtered_df)
print('')

# unique targets

df_ttd_drug_targets
max_target_row = df_ttd_drug_targets[df_ttd_drug_targets['TargetCount'] == 43]
UniqueTargets_ls = max_target_row['UniqueTargets'].tolist()
print('UniqueTargets_ls:', len(UniqueTargets_ls[0]))
print('')

# saving the DTIs to a file in a new directory

df = df_ttd_drug_target.copy()
shared_drugs = shared_filtered_df

df = df.rename(columns={'DrugName': 'Drug', 'TARGNAME': 'Targets'})
df['Drug'] = df['Drug'].str.lower()
filtered_df = df.loc[df['Drug'].isin(shared_drugs)].copy()

grouped_data = (
    filtered_df.groupby('Drug')['Targets']
    .apply(list)
    .reset_index()
)

# define a regular expression to match gene names in parentheses

gene_name_pattern = r"\([^)]*\)"

def extract_gene_names(target_list):
    """extracts and returns gene names within parentheses from a list."""
    gene_names = []
    for target in target_list:
        match = re.search(gene_name_pattern, target)
        if match:
            gene_name = match.group(0).strip("()")
            gene_names.append(gene_name)
    return gene_names

# apply the extraction function to the 'Targets' column

grouped_data['Targets'] = grouped_data['Targets'].apply(extract_gene_names)
output_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/Data2/DTIs/DTI_TTD.csv'
grouped_data.to_csv(output_path, index=False)

####################################################################################################

##### Database: Pocket_Features

import os
import pandas as pd

main_dir = '...'
os.chdir(os.path.join(main_dir, 'Data2/PocketFeatures/'))

# number of unique drugs

file_path = 'drug_gene_int_scores.txt'
df = pd.read_csv(file_path, delimiter='\t', header=None, names=['Drug', 'Targets', 'Score'])
old_drugbank_baseline = [drug.lower() for drug in old_drugbank_baseline]
drugs_list = df['Drug'].str.lower().unique().tolist()
print('# of drugs in PocketFeature list:', len(drugs_list))
print('')
common_drugs = [drug for drug in drugs_list if drug in old_drugbank_baseline]
print('# of drugs in PocketFeature list and the old DrugBank:', len(common_drugs))
print('drugs in PocketFeature list and the old DrugBank:', common_drugs)
print('')

# number of unique drugs

file_path = 'drug_gene_int_scores.txt'
df = pd.read_csv(file_path, delimiter='\t', header=None, names=['Drug', 'Targets', 'Score'])
old_drugbank_baseline = [drug.lower() for drug in old_drugbank_baseline]
drugs_list = df['Drug'].str.lower().unique().tolist()
print('# of drugs in PocketFeature list:', len(drugs_list))
print('')
common_drugs = [drug for drug in drugs_list if drug in old_drugbank_baseline]
print('# of drugs in PocketFeature list and the old DrugBank:', len(common_drugs))
print('drugs in PocketFeature list and the old DrugBank:', common_drugs)
print('')

# saving the DTIs to a file in a new directory

filtered_df = data[data['Drug'].isin(shared_drugs)].copy()
aggregated_data = filtered_df.groupby('Drug')['Targets'].apply(list).reset_index()
output_file_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/Data2/DTIs/DTI_PocketFeatures.csv'
aggregated_data.to_csv(output_file_path, index=False)

####################################################################################################
