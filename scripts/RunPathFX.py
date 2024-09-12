
import os

main_path = '/u/project/lune/malidoos/pathfxgen/'
os.chdir(os.path.join(main_path, 'pathfx_data_update/scripts/'))

# 29 shared drugs
drug_strings = ['acarbose', 'chlorthalidone', 'diflunisal', 'diphenhydramine', 'donepezil', 'dorzolamide', 'edrophonium', 'enalapril', 'galantamine', 'indapamide', 
'lisinopril', 'methazolamide', 'mexiletine', 'miglustat', 'pemetrexed', 'pentostatin', 'raloxifene', 'ramipril', 'rasagiline', 'rosuvastatin', 'saxagliptin', 
'sertraline', 'sitagliptin', 'sulfamethoxazole', 'trandolapril', 'tranexamic acid', 'triamterene', 'trimethoprim', 'vardenafil']

affixes = ['New_DrugBank', 'ChEMBL', 'PubChem', 'STITCH', 'TTD', 'Pocket_Features', 'all']

drug_ls = []
for drug in drug_strings:
    for affix in affixes:
        drug_ls.append(f'{drug}_{affix}')

analysis_name = 'PathFX_aim2_dti_run2'

for drug_name in drug_ls:
  cmd = 'python phenotype_enrichment_pathway_Pfxdti.py -d %s -a %s' %(drug_name, analysis_name)  
  os.system(cmd)

print('Ran PathFX!')
