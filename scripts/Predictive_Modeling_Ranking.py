
####################################################################################################

# Make the output_matrix (4 side effects) for Logistic Regression

import os
import pandas as pd

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/'))
res_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions/PathFX/results/pathfx_aim2_dti_run2/'

# Rows
all_metrics_path_df = pd.read_csv('all_metrics_pathfx_aim2_dti_run2.csv')
filtered_metrics_df = all_metrics_path_df[~all_metrics_path_df['Drug'].str.contains('Baseline')]
drug_database_names = filtered_metrics_df['Drug'].unique()

# Columns
file_ext = '_merged_neighborhood__assoc_table_.txt'
phenotypes = [
    'hypertension', 'prehypertension', 'hypertensive disease',
    'idiopathic pulmonary arterial hypertension', 'genetic hypertension',
    'pulmonary hypertension', 'essential hypertension',
    'hypertension, renovascular', 'idiopathic pulmonary hypertension',
    'renal hypertension', 'ocular hypertension',
    'hypertension, portal', 'phen_tp_hypertension',
    'phen_sig_tp_noFP_Dis_hypertension', 'phen_sig_tp_noFP_Sig_hypertension',

    'pancreatitis', 'acute pancreatitis', 'pancreatitis idiopathic',
    'carcinoma of pancreas', 'adenocarcinoma of pancreas',
    'pancreatitis, chronic', 'pancreatitis, alcoholic', 'phen_tp_pancreatitis',
    'phen_sig_tp_noFP_Dis_pancreatitis', 'phen_sig_tp_noFP_Sig_pancreatitis',

    'thrombocytopenia', 'thrombocytopenia 5', 'thrombocytopenia 6',
    'autoimmune thrombocytopenia', 'macrothrombocytopenia',
    'idiopathic thrombocytopenia', 'thrombocythemia, essential',
    'thrombocytopenia due to platelet alloimmunization',
    'phen_tp_thrombocytopenia', 'phen_sig_tp_noFP_Dis_thrombocytopenia',
    'phen_sig_tp_noFP_Sig_thrombocytopenia',

    'myocardial infarction', 'myocardial infarction 1',
    'myocardial infarction susceptibility to, 1 (finding)',
    'old myocardial infarction', 'myocardial ischemia',
    'acute myocardial infarction', 'myocardial failure',
    'phen_tp_myocardial infarction',
    'phen_sig_tp_noFP_Dis_myocardial infarction',
    'phen_sig_tp_noFP_Sig_myocardial infarction'
]

# Output_matrix
output_matrix = pd.DataFrame(0, index=drug_database_names, columns=phenotypes)
def check_phenotypes_in_files(res_path, drug_database_names, file_ext, phenotypes):
    for root, dirs, files in os.walk(res_path):
        for file_name in files:
            if file_ext in file_name:
                file_path_to_read = os.path.join(root, file_name)
                df_phen = pd.read_csv(file_path_to_read, sep='\t')
                phen_pred = df_phen['phenotype']
                drug_database = os.path.basename(root)
                if drug_database in drug_database_names:
                    for phenotype in phenotypes:
                        if phenotype in phen_pred.values:
                            output_matrix.loc[drug_database, phenotype] = 1
check_phenotypes_in_files(res_path, drug_database_names, file_ext, phenotypes)

os.chdir('predictive_model/')
output_matrix.to_csv('output_matrix3.csv', index=True)

# Drop the column with only 0 values

output_matrix = output_matrix.loc[:, (output_matrix != 0).any(axis=0)]
output_matrix.to_csv('output_matrix4.csv', index=True)
output_matrix

####################################################################################################

# Make the input_matrix for Logistic Regression

import os
import pandas as pd

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/'))

# Rows
all_metrics_path_df = pd.read_csv('all_metrics_pathfx_aim2_dti_run2.csv')
filtered_metrics_df = all_metrics_path_df[~all_metrics_path_df['Drug'].str.contains('Baseline')]
drug_database_names = filtered_metrics_df['Drug'].unique()

# Columns
shared_drugs_targets_df = pd.read_csv('shared_drugs_targets.csv')
all_targets = set()
for targets in shared_drugs_targets_df['Union of All Targets']:
    target_list = eval(targets)
    all_targets.update(target_list)
unique_targets = list(all_targets)

# Input_matrix
input_matrix = pd.DataFrame(0, index=drug_database_names, columns=unique_targets)
for _, row in shared_drugs_targets_df.iterrows():
    drug = row['Drug']
    for database in row['Databases'].split(', '):
        drug_database = f"{drug}_{database}"
        if drug_database in drug_database_names:
            target_list = eval(row['Union of All Targets'])
            input_matrix.loc[drug_database, target_list] = 1

os.chdir('predictive_model/')
input_matrix.to_csv('input_matrix1.csv', index=True)

# Remove columns where the sum is greater than 25

column_sums = input_matrix.sum()
input_matrix = input_matrix.loc[:, column_sums >= 25]
input_matrix.to_csv('input_matrix2.csv', index=True)

####################################################################################################

# Develop a predictive model to identify the most important targets and databases in predicting the phenotypes

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

print('')
print('Predictive modeling ...\n')

main_path = '/u/project/lune/malidoos/aim2/'
os.chdir(os.path.join(main_path, 'predictive_model/'))

print('Read the data and split ...\n')

input_matrix = pd.read_csv('input_matrix2.csv', index_col=0)
output_matrix = pd.read_csv('output_matrix4.csv', index_col=0)

print(f'Input matrix shape: {input_matrix.shape}')
print(f'Output matrix shape: {output_matrix.shape}\n')

accuracy_scores_dict = {}
best_rows_dict = {}
coefficients_dict = {}
ranked_coefficients_dict = {}
all_coefficients_dict = {}

for column in output_matrix.columns:
    print(f'Processing phenotype: {column}\n')

    y = output_matrix[column]

    print('Apply Logistic Regression ...\n')

    logistic = LogisticRegression(penalty='l1', solver='liblinear')
    logistic.fit(input_matrix, y)

    print('Evaluate ...\n')

    y_pred = logistic.predict(input_matrix)
    accuracy = accuracy_score(y, y_pred)
    accuracy_scores_dict[column] = accuracy
    print(f'Accuracy Score for {column}: {accuracy}\n')

    coefficients = pd.Series(logistic.coef_.flatten(), index=input_matrix.columns)
    coefficients_dict[column] = coefficients
    all_coefficients_dict[column] = coefficients 

    ranked_coefficients = coefficients.abs().sort_values(ascending=False)
    ranked_coefficients_dict[column] = ranked_coefficients

    residuals = np.abs(y - y_pred)
    best_rows = residuals.sort_values().index[:len(residuals) // 2]
    best_rows_dict[column] = best_rows
    print(f'Best Input Rows (Databases) for {column}: {best_rows}\n')

coefficients_df = pd.DataFrame(coefficients_dict).T
coefficients_df.to_csv('coefficients.csv')

ranked_coefficients_df = pd.DataFrame(ranked_coefficients_dict).T
ranked_coefficients_df.to_csv('ranked_coefficients_with_scores.csv')

all_coefficients_df = pd.DataFrame(all_coefficients_dict).T
all_coefficients_df.to_csv('all_coefficients_with_signs.csv')

pd.DataFrame.from_dict(accuracy_scores_dict, orient='index', columns=['Accuracy']).to_csv('accuracy_scores.csv', index=True)
pd.DataFrame.from_dict(best_rows_dict, orient='index').to_csv('best_rows.csv', index=True)

print('Accuracy scores, best rows, coefficients, ranked coefficients with scores, and all coefficients with signs have been saved.')

####################################################################################################

# Plot the positive and negative coefficients and visualize their distribution

import os
import pandas as pd
import matplotlib.pyplot as plt

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/predictive_model/'))

coefficients_df = pd.read_csv('predictive_model4- all_coefficients_with_signs.csv', index_col=0)

for phenotype in coefficients_df.index:

    coefficients = coefficients_df.loc[phenotype]
    coefficients_sorted = coefficients.sort_values()
    significant_coefficients = coefficients_sorted[coefficients_sorted != 0]
    num_zeros_to_display = 5 
    placeholders = ['...'] * num_zeros_to_display  
    x_labels = significant_coefficients.index.tolist() + placeholders
    y_values = significant_coefficients.tolist() + [0] * num_zeros_to_display

    plt.figure(figsize=(8, 4))
    plt.bar(x_labels, y_values, color=['red' if x < 0 else 'blue' for x in y_values])
    plt.title(f'Coefficients for {phenotype}')
    plt.xlabel('Targets')
    plt.ylabel('Coefficient Value')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_dir = 'coefficients_plot'
    plt.savefig(os.path.join(output_dir, f'coefficients_plot_{phenotype}.tiff'), format='tiff', dpi=300)

    plt.show()

####################################################################################################

# Compare the identified targets with the drug targets per database and take targets with +&- coefficients

import os
import pandas as pd

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/predictive_model/'))

file_path = 'predictive_model4- all_coefficients_with_signs.csv'
ranked_targets_with_scores = pd.read_csv(file_path, index_col=0)

shared_drugs_targets_path = 'shared_drugs_targets.csv'
shared_drugs_targets = pd.read_csv(shared_drugs_targets_path)

comparison_results = []

for index, row in shared_drugs_targets.iterrows():
    drug = row['Drug']
    databases = row['Databases'].split(',')

    db_target_counts = {db: 0 for db in databases}
    db_target_percentages = {db: 0 for db in databases}
    db_identified_counts = {f'{db.strip()} Identified Targets': 0 for db in databases}
    db_total_counts = {f'{db.strip()} Total Targets': 0 for db in databases}

    all_targets = []
    for db in databases:
        targets = row[f'{db.strip()} Targets']
        if isinstance(targets, str):
            targets = targets.strip("[]").replace("'", "").split(', ')
            all_targets.extend(targets)
            db_target_counts[db] = len(targets)

    all_targets_set = set(all_targets)

    for phenotype in ranked_targets_with_scores.index:
        ranked_targets = set(ranked_targets_with_scores.loc[phenotype][ranked_targets_with_scores.loc[phenotype] != 0].dropna().index)
        intersection = ranked_targets.intersection(all_targets_set)
        percentage = (len(intersection) / len(ranked_targets)) * 100 if ranked_targets else 0

        for db in databases:
            db_targets = set(row[f'{db.strip()} Targets'].strip("[]").replace("'", "").split(', '))
            db_intersection = ranked_targets.intersection(db_targets)
            db_percentage = (len(db_intersection) / len(db_targets)) * 100 if db_targets else 0
            db_target_percentages[db] = round(db_percentage, 2)
            db_identified_counts[f'{db.strip()} Identified Targets'] = len(db_intersection)
            db_total_counts[f'{db.strip()} Total Targets'] = len(db_targets)

        comparison_results.append({
            'Drug': drug,
            'Phenotype': phenotype,
            'Database': ','.join(databases),
            'Ranked Targets': len(ranked_targets),
            'Common Targets': len(intersection),
            'Percentage': round(percentage, 2),
            **db_total_counts,
            **db_identified_counts,
            **db_target_percentages
        })

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('comparison_drugs_phenotypes_databases_targets.csv', index=False)

####################################################################################################

# Rank databases per phenotype based on target percentages

import os
import pandas as pd

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/predictive_model/'))

comparison_file_path = 'comparison_drugs_phenotypes_databases_targets2.csv'
comparison_df = pd.read_csv(comparison_file_path)

database_rankings = {}

percentage_columns = comparison_df.columns[-6:]

for phenotype in comparison_df['Phenotype'].unique():

    phenotype_data = comparison_df[comparison_df['Phenotype'] == phenotype]

    db_percentages = {}

    for col in percentage_columns:
        db_name = col.replace(' Percentage', '')
        mean_percentage = phenotype_data[col].mean() 

        db_percentages[db_name] = mean_percentage

    ranked_databases = sorted(db_percentages.items(), key=lambda item: item[1], reverse=True)

    database_rankings[phenotype] = ranked_databases

ranked_df = pd.DataFrame.from_dict(database_rankings, orient='index')

new_ranked_column_names = {
    0: '1st_Ranked_Database',
    1: '2nd_Ranked_Database',
    2: '3rd_Ranked_Database',
    3: '4th_Ranked_Database',
    4: '5th_Ranked_Database',
    5: '6th_Ranked_Database'
}

ranked_df.rename(columns=new_ranked_column_names, inplace=True)
ranked_df.to_csv('ranked_databases_per_phenotype_based_on_percentages.csv', index=True)

####################################################################################################

# A heatmap showing the ranking across all phenotypes in a single plot
# Each cell in the heatmap would represent the rank of a specific database for a specific phenotype

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/predictive_model/'))

ranked_df = pd.read_csv('ranked_databases_per_phenotype_based_on_percentages.csv', index_col=0)

def extract_numeric(value):
    return float(value.split(', ')[1].replace(')', ''))

def extract_db_name(value):
    return value.split(',')[0].replace("('", "").replace("'", "").strip()

numeric_df = ranked_df.copy()
for col in numeric_df.columns:
    numeric_df[col] = numeric_df[col].map(extract_numeric)

database_names = [extract_db_name(value) for value in ranked_df.iloc[0]]
numeric_df.columns = database_names

cmap = sns.diverging_palette(20, 250, n=6, as_cmap=True)  

plt.figure(figsize=(12, 10))
ax = sns.heatmap(numeric_df, cmap=cmap, annot=True, fmt=".1f", linewidths=.5, cbar_kws={'label': 'Averaged Target Percentage'},
                 yticklabels=numeric_df.index, xticklabels=numeric_df.columns)

plt.title('Heatmap of Database Rankings per Phenotype')
plt.xlabel('Databases')
plt.ylabel('Phenotypes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('heatmap_database_rankings_averaged target percentage.tiff', format='tiff', dpi=300)
plt.show()

####################################################################################################
