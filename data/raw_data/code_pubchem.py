####################

# libraries

import os
import pandas as pd

# directory

data_dir = '.../aim2/data2/pubchem'
os.chdir(data_dir)

print('')
print('####################')
print('')

####################

folder_path = data_dir
final_data = pd.DataFrame(columns=['cid', 'cmpdname', 'genename'])

for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, low_memory=False)

        if not df.empty:
            cid = df['cid'].iloc[0] if 'cid' in df.columns else None
            print('cid: ', cid)
            
            cmpdname = df['cmpdname'].iloc[0] if 'cmpdname' in df.columns else None

            genenames = df['genename'].dropna().unique().tolist() if 'genename' in df.columns else None

            final_data = final_data.append({'cid': cid, 'cmpdname': cmpdname, 'genename': genenames}, ignore_index=True)

final_data.to_csv('pubchem_output.csv', index=False)
print('Done!')

print('')
print('####################')
print('')

####################
