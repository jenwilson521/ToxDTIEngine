
####################################################################################################

# Combine all selected drug-target interactions across all six databases
# Include the database names to the drug names and make drug_all rows as well

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

combined_data = {}
all_targets_data = {}

for db_name, file_path in files.items():
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        drug = row['Drug']
        targets = eval(row['Targets'])
        drug_db_entry = f"{drug}_{db_name}"

        if drug_db_entry not in combined_data:
            combined_data[drug_db_entry] = {'drug': drug, 'database': db_name, 'targets': set(targets)}
        else:
            combined_data[drug_db_entry]['targets'].update(targets)

        if drug not in all_targets_data:
            all_targets_data[drug] = set(targets)
        else:
            all_targets_data[drug].update(targets)

data = []
for entry, details in combined_data.items():
    data.append((details['drug'], list(details['targets']), f"{details['drug']}_{details['database']}"))

for drug, all_targets in all_targets_data.items():
    data.append((drug, list(all_targets), f"{drug}_all"))

combined_df = pd.DataFrame(data, columns=['Drug', 'Targets', 'Drug_Database'])

output_file = os.path.join(main_dir, 'Data2/DTIs/DTIs_filtered/combined_ddtis.csv')
combined_df.to_csv(output_file, index=False)

####################################################################################################

# Pipeline to evaluate PathFX predictions

# Make the phenotype and cui lists including only the label side effects

import os
import pickle

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/rscs/'))

phene_ls = ['edema', 'gastric ulcer', 'neuroleptic malignant syndrome', 'delirium',
            'hyperlipidemia', 'completed suicide', 'hepatic necrosis',
            'tardive dyskinesia', 'proteinuria', 'hypertension', 'hemorrhage',
            'myocardial infarction', 'deep vein thrombosis', 'sepsis', 'cardiac arrest',
            'thrombocytopenia', 'agranulocytosis', 'stevens-johnson syndrome',
            'cerebral infarction', 'pancreatitis', 'peripheral neuropathy',
            'pulmonary edema', 'myopathy', 'pneumonia', 'anaphylaxis',
            'seizures', 'lung cyst', 'anemia', 'tachycardia',
            'prolonged qt interval', 'sleep disorders', 'sleep apnea syndromes']

p2c = pickle.load(open('Pfx050120_all_phens_to_cuis.pkl','rb'))
name_ls = [name.lower() for name, value in p2c.items() if name.lower() in phene_ls]
cui_ls = [value for name, value in p2c.items() if name.lower() in phene_ls]

# Map all phenotypes' cuis to the side effects' cuis

map_to_orig_cui_dict = {

  'C0040034': 'C0040034',
  'C4310789': 'C0040034',
  'C4015537': 'C0040034',
  'C0040028': 'C0040034',
  'C0242584': 'C0040034',
  'C0920163': 'C0040034',
  'C2751260': 'C0040034',
  'C0272286': 'C0040034',
  'C0038358': 'C0038358',
  'C0030920': 'C0038358',
  'C0013604': 'C0013604',
  'C1527311': 'C0013604',
  'C0686347': 'C0686347',
  'C3714760': 'C0686347',
  'C0454606': 'C0686347',
  'C0020473': 'C0020473',
  'C0020445': 'C0020473',
  'C0020557': 'C0020473',
  'C0020443': 'C0020473',
  'C0745103': 'C0020473',
  'C0027051': 'C0027051',
  'C0155668': 'C0027051',
  'C1832662': 'C0027051',
  'C0151744': 'C0027051',
  'C0155626': 'C0027051',
  'C1959583': 'C0027051',
  'C0032285': 'C0032285',
  'C0032241': 'C0032285',
  'C0032300': 'C0032285',
  'C1535939': 'C0032285',
  'C0155862': 'C0032285',
  'C0001824': 'C0001824',
  'C1282609': 'C0001824',
  'C0038325': 'C0038325',
  'C3658302': 'C0038325',
  'C1274933': 'C0038325',
  'C3658301': 'C0038325',
  'C0034063': 'C0034063',
  'C0848538': 'C0034063',
  'C0243026': 'C0243026',
  'C0456103': 'C0243026',
  'C0036690': 'C0243026',
  'C0018790': 'C0018790',
  'C3826614': 'C0018790',
  'C1720824': 'C0018790',
  'C0149871': 'C0149871',
  'C0040053': 'C0149871',
  'C0087086': 'C0149871',
  'C0836924': 'C0149871',
  'C2712843': 'C0149871',
  'C0042487': 'C0149871',
  'C0740376': 'C0149871',
  'C3278737': 'C3278737',
  'C1696708': 'C3278737',
  'C3203102': 'C3278737',
  'C0085580': 'C3278737',
  'C0020542': 'C3278737',
  'C0020545': 'C3278737',
  'C0152171': 'C3278737',
  'C0028840': 'C3278737',
  'C0020541': 'C3278737',
  'C0598428': 'C3278737',
  'C0020544': 'C3278737',
  'C0020538': 'C3278737',
  'C0026848': 'C0026848',
  'C1853926': 'C0026848',
  'C1850718': 'C0026848',
  'C2678065': 'C0026848',
  'C0175709': 'C0026848',
  'C0410207': 'C0026848',
  'C0878544': 'C0026848',
  'C0751713': 'C0026848',
  'C0033687': 'C0033687',
  'C4022832': 'C0033687',
  'C0019080': 'C0019080',
  'C0852361': 'C0019080',
  'C0031117': 'C0031117',
  'C1263857': 'C0031117',
  'C0235025': 'C0031117',
  'C0149940': 'C0031117',
  'C0442874': 'C0031117',
  'C0030305': 'C0030305',
  'C0747198': 'C0030305',
  'C0001339': 'C0030305',
  'C0376670': 'C0030305',
  'C0149521': 'C0030305',
  'C0235974': 'C0030305',
  'C0279176': 'C0030305',
  'C0007785': 'C0007785',
  'C0751955': 'C0007785',
  'C0751956': 'C0007785',
  'C0038454': 'C0007785',
  'C0751846': 'C0007785',
  'C0751847': 'C0007785',
  'C0751849': 'C0007785',
  'C0740391': 'C0007785',
  'C2937358': 'C0007785',
  'C0007786': 'C0007785',

  'C0027849': 'C0027849', #1
  'C0011206': 'C0011206', #1
  'C0852733': 'C0852733', #1
  'C0151798': 'C0151798', #1

  'C0039231': 'C0039231', #0
  'C0080203': 'C0039231', #0
  'C0151878': 'C0151878', #0
  'C0002792': 'C0002792', #0
  'C0850803': 'C0002792', #0
  'C0036572': 'C0036572', #0
  'C3809174': 'C0036572', #0
  'C0751494': 'C0036572', #0
  'C0234535': 'C0036572', #0
  'C0494475': 'C0036572', #0
  'C0234533': 'C0036572', #0
  'C0546483': 'C0546483', #0
  'C0002871': 'C0002871', #0
  'C0002873': 'C0002871', #0
  'C0037315': 'C0037315', #0
  'C0851578': 'C0851578', #0
  'C4042891': 'C0851578', #0
  'C0037317': 'C0851578', #0
  'C0917801': 'C0851578'  #0

}

def map_to_orig_cui(pred):

  orig_cui = map_to_orig_cui_dict[pred]

  return orig_cui

# The drug toxicity data is considered as the truth

def evaluate_drug_pathfx_analysis(drug_given_name, analysis_name, cui_ls):

  '''
  The inputs of this function are the drug name (string) and the list of CUIs.
  This function runs PathFX and evaluates the predictions.
  '''

  import os
  import pandas as pd
  import pickle

  main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
  os.chdir(main_path)

  # Read the drug toxicity data and count the # of side effects for a drug

  dft = pd.read_csv('Drugs_labeled_for_AEs.txt', sep='\t', low_memory=False)
  drug_name = drug_given_name.split('_')[0]
  count = 0
  phen_label = []
  for i in (dft.columns)[:]:
    dn = dft[i].str.lower()
    drugcols = dft[(dn==drug_name.lower())]
    count = len(drugcols) + count
    if len(drugcols) !=0:
      phen_label.append(i.lower())

  # Convert phenotypes to CUI terms

  os.chdir(os.path.join(main_path, 'PathFX/rscs/'))
  p2c = pickle.load(open('Pfx050120_all_phens_to_cuis.pkl','rb'))
  name_label = [name.lower() for name, value in p2c.items() if name.lower() in phen_label]
  cui_label = [value for name, value in p2c.items() if name.lower() in phen_label]

  # Read the PathFX prediction table

  res_path = os.path.join(main_path, 'PathFX/results/', analysis_name, drug_given_name)
  file_ext = '_merged_neighborhood__assoc_table_.txt'
  df_drug = None
  for root, dirs, files in os.walk(res_path):
    for file_name in files:
      if file_ext in file_name:
        file_path_to_read = os.path.join(root, file_name)
        df_drug = pd.read_csv(file_path_to_read, sep='\t')
        cui_pred = df_drug['cui']
        break
    if df_drug is not None:
      break

  if df_drug is None:
    print(f'No file found with extension {file_ext} in {res_path}')
    return

  # Results: TP & FP & FN & TN.

  print('')
  print('results: confusion matrix')
  print('')

  cui_predls_all = [x for x in cui_pred if x in cui_ls]

  cui_predls_ = []
  for pred in cui_predls_all:
    new_pred = map_to_orig_cui(pred)
    cui_predls_.append(new_pred)
  cui_predls = list(set(cui_predls_))

  intersection_tp = set(cui_predls).intersection(cui_label)
  cui_tp_ls = list(intersection_tp)
  cui_tp = len(cui_tp_ls)
  print('TPs:', cui_tp)
  print('TP cuis:', cui_tp_ls)
  print('')

  cui_fp_ls = [x for x in cui_predls if x not in cui_tp_ls]
  cui_fp = len(cui_fp_ls)
  print('FPs:', cui_fp)
  print('FP cuis:', cui_fp_ls)
  print('')

  cui_fn_ls = [x for x in cui_label if x not in cui_tp_ls]
  cui_fn = len(cui_fn_ls)
  print('FNs:', cui_fn)
  print('FN cuis:', cui_fn_ls)
  print('')

  cui_predlabel = list(set(cui_predls + cui_label))

  cui_tn_ls = [x for x in cui_ls if x not in cui_predlabel]
  cui_tn = len(cui_tn_ls)
  print('TNs:', cui_tn)
  print('TN cuis:', cui_tn_ls)
  print('')

  # Results: sensitivity & specificity & precision.

  print('results: metrics')
  print('')

  TP = cui_tp
  FP = cui_fp
  FN = cui_fn
  TN = cui_tn

  conf_sensitivity = (TP / float(TP + FN + 0.00000001))
  conf_specificity = (TN / float(TN + FP + 0.00000001))
  conf_precision = (TP / float(TP + FP + 0.00000001))

  print(f'Sensitivity: {round(conf_sensitivity,2)}')
  print(f'Specificity: {round(conf_specificity,2)}')
  print(f'Precision: {round(conf_precision,2)}')

  # Results: table of results.

  header = ('Drug', 'TP', 'TN', 'FP', 'FN', 'Sensitivity',
          'Specificity', 'Precision')
  rows = []
  met = [drug_given_name, TP, TN, FP, FN, conf_sensitivity, conf_specificity, conf_precision]
  rows.append(met)
  dfmet = pd.DataFrame.from_records(rows, columns=header)
  os.chdir(os.path.join(main_path, 'PathFX/results/'))
  dfmet.to_csv('./metrics.csv', index=False)

  return

# Call the function

# Run the evaluation function for multiple drugs and combine all results into one CSV file

# 29 shared drugs
drug_strings = ['acarbose', 'chlorthalidone', 'diflunisal', 'diphenhydramine',
                'donepezil', 'dorzolamide', 'edrophonium', 'enalapril',
                'galantamine', 'indapamide', 'lisinopril', 'methazolamide',
                'mexiletine', 'miglustat', 'pemetrexed', 'pentostatin',
                'raloxifene', 'ramipril', 'rasagiline', 'rosuvastatin',
                'saxagliptin', 'sertraline', 'sitagliptin', 'sulfamethoxazole',
                'trandolapril', 'tranexamic acid', 'triamterene', 'trimethoprim',
                'vardenafil']

affixes = ['New_DrugBank', 'ChEMBL', 'PubChem', 'STITCH', 'TTD',
           'Pocket_Features', 'all']

drug_ls = []
for drug in drug_strings:
    for affix in affixes:
        drug_ls.append(f'{drug}_{affix}')

analysis_name = 'pathfx_aim2_dti_run2'

import os
import pandas as pd
from csv import reader

labels = ('Drug', 'TP', 'TN', 'FP', 'FN', 'Sensitivity', 'Specificity', 'Precision')
rows = []

for drug in drug_ls:
  evaluate_drug_pathfx_analysis(drug, analysis_name, cui_ls)
  os.chdir(os.path.join(main_path, 'PathFX/results/'))
  with open('metrics.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    if header != None:
      for row in csv_reader:
        rows.append(row)

  print('')
  print('')
  print('')

dfmetall = pd.DataFrame.from_records(rows, columns=labels)
os.chdir(os.path.join(main_path, 'PathFX/results/'))
dfmetall.to_csv('./all_metrics.csv')

####################################################################################################

# Evaluation plots

import os
import pandas as pd
import matplotlib.pyplot as plt

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/'))

df = pd.read_csv('all_metrics_pathfx_aim2_dti_run2.csv')

base_drugs = df['Drug'].str.split('_').str[0].unique()

markers = ['D', 'o', 'P', '*', 'X', 's', '^', 'v']
colors = ['b', 'r', 'orange', 'k', 'c', 'g', 'm', 'y']

for base_drug in base_drugs:
    fig, ax = plt.subplots()

    drug_data = df[df['Drug'].str.startswith(base_drug)]
    sub_drugs = drug_data['Drug'].unique()

    for i, sub_drug in enumerate(sub_drugs):
        sub_drug_data = drug_data[drug_data['Drug'] == sub_drug]
        x = sub_drug_data['Specificity']
        y = sub_drug_data['Sensitivity']

        ax.scatter(x, y, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=sub_drug)

    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.legend(title='Sub-categories', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title_fontsize='small', frameon=True)
    ax.set_title(f'Evaluation metrics for {base_drug.title()}')

    os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/evaluation_plots/'))
    file_base_name = f'Evaluation_Plot_{base_drug.title()}'
    fig.savefig(f'{file_base_name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{file_base_name}.tiff', dpi=300, bbox_inches='tight', format='tiff')

    plt.show()
    plt.close(fig)

####################################################################################################

# Plot the difference in sensitivity between the baseline and the other points

import os
import pandas as pd
import matplotlib.pyplot as plt

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/'))

df = pd.read_csv('all_metrics_pathfx_aim2_dti_run2.csv')

base_drugs = df['Drug'].str.split('_').str[0].unique()

markers = ['D', 'h', 'P', '*', 'X', 's', '^', 'v']
colors = ['b', 'g', 'orange', 'k', 'c', 'g', 'm', 'y']

for base_drug in base_drugs:
    fig, ax = plt.subplots()

    drug_data = df[df['Drug'].str.startswith(base_drug)]
    baseline_data = drug_data[drug_data['Drug'] == f"{base_drug}_Baseline"]

    if baseline_data.empty:
        continue

    baseline_sensitivity = baseline_data['Sensitivity'].values[0]

    sub_drugs = drug_data['Drug'].unique()

    ax.scatter([baseline_sensitivity], [0], marker='o', color='red', label=f"{base_drug}_Baseline")

    for i, sub_drug in enumerate(sub_drugs):
        if sub_drug == f"{base_drug}_Baseline":
            continue
        sub_drug_data = drug_data[drug_data['Drug'] == sub_drug]
        sub_drug_sensitivity = sub_drug_data['Sensitivity'].values[0]
        sensitivity_diff = sub_drug_sensitivity - baseline_sensitivity

        ax.scatter([sub_drug_sensitivity], [0], marker=markers[i % len(markers)], color=colors[i % len(colors)], label=sub_drug)

    ax.axvline(baseline_sensitivity, color='gray', linestyle='--', linewidth=0.5)
    ax.set_yticks([])
    ax.set_xlabel('Sensitivity')
    ax.legend(title='Sub-categories', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title_fontsize='small', frameon=True)
    ax.set_title(f'Sensitivity differences compared to baseline for {base_drug.title()}')

    os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/sensitivity_plots/'))
    file_base_name = f'Sensitivity_Plot_{base_drug.title()}'
    fig.savefig(f'{file_base_name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{file_base_name}.tiff', dpi=300, bbox_inches='tight', format='tiff')

    plt.show()
    plt.close(fig)

####################################################################################################

# Plot the difference in specificity between the baseline and the other points

import os
import pandas as pd
import matplotlib.pyplot as plt

main_path = '/content/gdrive/MyDrive/PhD_Lab/Project_Drug_Toxicity_Network_Predictions'
os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/'))

df = pd.read_csv('all_metrics_pathfx_aim2_dti_run2.csv')

base_drugs = df['Drug'].str.split('_').str[0].unique()

markers = ['D', 'h', 'P', '*', 'X', 's', '^', 'v']
colors = ['b', 'g', 'orange', 'k', 'c', 'g', 'm', 'y']

for base_drug in base_drugs:
    fig, ax = plt.subplots()

    drug_data = df[df['Drug'].str.startswith(base_drug)]
    baseline_data = drug_data[drug_data['Drug'] == f"{base_drug}_Baseline"]

    if baseline_data.empty:
        continue

    baseline_specificity = baseline_data['Specificity'].values[0]

    sub_drugs = drug_data['Drug'].unique()

    ax.scatter([baseline_specificity], [0], marker='o', color='red', label=f"{base_drug}_Baseline")

    for i, sub_drug in enumerate(sub_drugs):
        if sub_drug == f"{base_drug}_Baseline":
            continue
        sub_drug_data = drug_data[drug_data['Drug'] == sub_drug]
        sub_drug_specificity = sub_drug_data['Specificity'].values[0]
        specificity_diff = sub_drug_specificity - baseline_specificity

        ax.scatter([sub_drug_specificity], [0], marker=markers[i % len(markers)], color=colors[i % len(colors)], label=sub_drug)

    ax.axvline(baseline_specificity, color='gray', linestyle='--', linewidth=0.5)
    ax.set_yticks([])
    ax.set_xlabel('Specificity')
    ax.legend(title='Sub-categories', loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title_fontsize='small', frameon=True)
    ax.set_title(f'Specificity differences compared to baseline for {base_drug.title()}')

    os.chdir(os.path.join(main_path, 'PathFX/results/pathfx_aim2_dti_run2_Evaluation/specificity_plots/'))
    file_base_name = f'Specificity_Plot_{base_drug.title()}'
    fig.savefig(f'{file_base_name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{file_base_name}.tiff', dpi=300, bbox_inches='tight', format='tiff')

    plt.show()
    plt.close(fig)

####################################################################################################
