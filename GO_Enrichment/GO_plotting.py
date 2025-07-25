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

all_rows = []
for f in rev_f:
	dbname = os.path.split(f)[-1].replace("_Revigo_MF_OnScreenTable.tsv","")
	print(dbname)
	row_data = {'Database':dbname}
	df = pd.read_csv(f,delimiter='\t',header=0,names=column_names)
	for (gon,gov) in zip(df.Name,df.Value):
		if gon in common_names:
			continue
		if len(row_data) < 11: #
			row_data[gon]=-gov 
	all_rows.append(row_data)


go_df = pd.DataFrame(all_rows).set_index('Database')
plot_df = go_df.T.fillna(0)

fig,ax = plt.subplots()
g = sns.clustermap(plot_df, cmap='rocket', center=0, annot=True, fmt='.0f')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.subplots_adjust(right=0.3,left=0.12,bottom=0.2)
g.ax_cbar.set_position((0.05, .3, .03, .4))
g.ax_heatmap.set_ylabel('GO Term')
plt.savefig(os.path.join(rdir,'GO_enrichment_10topUnique.png'),format='png',dpi=300)
