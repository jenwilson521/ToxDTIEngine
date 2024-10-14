####################

# libraries
import os
import sqlite3
import pandas as pd

# directory
data_dir = '/.../chembl_33/chembl_33_sqlite/'
os.chdir(data_dir)

print('')
print('####################')
print('')

####################

# specify the path to the SQLite database file
database_file = 'chembl_33.db'
database_path = os.path.join(data_dir, database_file)

# connect to the ChEMBL SQLite database
conn = sqlite3.connect(database_path)

# modify the WHERE clause for multiple IDs
# read the target_ChEMBL_ID list

file_path = 'target_ChEMBL_ID.txt'
with open(file_path, 'r') as file:
    target_chEMBL_ID_list = [line.strip() for line in file.readlines()]
    
chembl_ids = target_chEMBL_ID_list

# query
query = f"""
    SELECT
        td.chembl_ID as target_chembl_ID,
        td.organism,
        td.tax_ID,
        td.pref_name,
        cs.component_synonym
    FROM
        target_dictionary td
        LEFT JOIN target_components tc ON td.tid=tc.tid
        LEFT JOIN component_synonyms cs ON tc.component_ID=cs.component_ID        
    WHERE
        syn_type = 'GENE_SYMBOL'
        AND chembl_ID IN ({', '.join(map(lambda x: f"'{x}'", chembl_ids))})
"""

# fetch the results into a dataframe
df = pd.read_sql_query(query, conn)

# display the result dataframe
print(df.head())

# close the connection
conn.close()

####################
