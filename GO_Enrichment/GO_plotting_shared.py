import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

rdir = '../results/dti_revisions/'
column_names = ['TermID','Name','Value','LogSize','Frequency','Uniqueness','Dispensability','Representative']
rev_f = [os.path.join(rdir,f) for f in os.listdir(rdir) if 'Revigo_MF_OnScreenTable.tsv' in f]

name_sets = []
for f in rev_f:
    df = pd.read_csv(f, sep='\t', header=0, names=column_names)
    name_sets.append(set(df['Name'].dropna()))

common_names = set.intersection(*name_sets)

value_dict = {}
for f in rev_f:
    dbname = os.path.split(f)[-1].replace("_Revigo_MF_OnScreenTable.tsv","")
    print(dbname)
    df = pd.read_csv(f,delimiter='\t',header=0,names=column_names)
    df_filtered = df[df['Name'].isin(common_names)]
    row_data = {'Database':dbname}
    name_value_map = dict(zip(df_filtered['Name'], -df_filtered['Value']))
    value_dict[dbname] = name_value_map

heatmap_df = pd.DataFrame.from_dict(value_dict, orient='index').T
heatmap_df = heatmap_df.sort_index().sort_index(axis=1)  
heatmap_df.index.names = ['Database']

fig,ax = plt.subplots()
g = sns.clustermap(heatmap_df, cmap='coolwarm', center=0, annot=True, fmt='.0f')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.subplots_adjust(right=0.3,left=0.12,bottom=0.2)
g.ax_cbar.set_position((0.05, .3, .03, .4))
g.ax_heatmap.set_ylabel('GO Term')
g.ax_heatmap.set_xlabel('Database')
plt.savefig(os.path.join(rdir,'GO_enrichment_COMMON.png'),format='png',dpi=300)
