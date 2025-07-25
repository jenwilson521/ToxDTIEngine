# written to do some formatting for Mohamad's revisions
# written 7-24-25 JLW

import pickle,csv,os,math
import pandas as pd
from collections import defaultdict

# results directory
rdir = '../results/dti_revisions'
	
# count common protein familes across datasets
column_names = ['GeneID', 'MappedID', 'GeneName', 'subFam','proteinClass','organism']
pan_f = [os.path.join(rdir,f) for f in os.listdir(rdir) if 'PANTHER.txt' in f]
all_rows = []
for f in pan_f:
#	dbname = f.strip('.txt').split('_')[-1]
	dbname = os.path.split(f)[-1].replace("_PANTHER.txt","")
	print(dbname)
	pcdf = pd.read_csv(f,delimiter='\t',names=column_names)
	row_data = {'Database':dbname}
	pc_counts = defaultdict(int)
	for pc in pcdf.proteinClass:
		if type(pc)!=type('str') and math.isnan(float(pc)): # some are not mapped to a class
			continue
		pc_counts[pc]+=1
	top_counts = sorted([x for x in pc_counts.items()], key = lambda x:x[1],reverse=True)[0:10]
	for (pc,pc_count) in top_counts:
		row_data[pc]=pc_count
	all_rows.append(row_data)

# plot using heatmap
import seaborn as sns
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

pc_df = pd.DataFrame(all_rows)
pc_df = pc_df.fillna(0)
pc_df = pc_df.set_index('Database')
plot_df = pc_df.T

fig,ax = plt.subplots()
g = sns.clustermap(plot_df)
plt.subplots_adjust(right=0.6,left=0.2)
g.ax_cbar.set_position((0.1, .3, .03, .4))
ax.tick_params(axis='x', rotation=45)
plt.savefig(os.path.join(rdir,'panther_pcs_perDB.png'),format='png',dpi=300)

