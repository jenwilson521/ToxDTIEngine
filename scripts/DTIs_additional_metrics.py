# written 10-28-25 JLW
# to assess some additional metrics
# for DTI revisions

import pickle,os,csv
import pandas as pd
from collections import defaultdict

rdir ='/Users/jenniferwilson/Documents/UCLA/drafts_papers_theses/Mohamad_DTIs/resubmission_2/' 
f = os.path.join(rdir,'SM 6- all_metrics_pathfx_dtis_v2.csv')
df = pd.read_csv(f)

dti_sources =['All_Targets', 'Baseline', 'ChEMBL', 'New_DrugBank', 'Pocket_Features', 'PubChem', 'STITCH', 'TTD']

all_mcc = defaultdict(list)
all_fone = defaultdict(list)

for dsource in dti_sources:
	df_short = df[df.Drug.str.contains(dsource)]
	for (d,fone,mccv) in zip(df_short.Drug,df_short['F1 Score'].to_list(),df_short.MCC):
		all_mcc[dsource].append(float(mccv))
		all_fone[dsource].append(float(fone))

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)

w = 0.8
x = range(len(all_mcc))
edge_colors = ['blue','red','orange','black','cyan','green','magenta','gold'] # to match other score plots

fig,ax = plt.subplots(1,2,figsize=(10,4))
# first MCC scores
(labels_mcc,y_mcc) = zip(*sorted(all_mcc.items())) 
ax[0].bar(x,
	height=[np.mean(yi) for yi in y_mcc],
	yerr=[np.std(yi) for yi in y_mcc],    # error bars
	capsize=12, # error bar cap width in points
	width=w,    # bar width
	tick_label=labels_mcc,
	#color=(0,0,0,0,0,0,0,0),  # face color transparent
	color=('w','w','w','w','w','w','w','w'),  # face color transparent
	edgecolor=edge_colors,)

for i in range(len(x)):
	# distribute scatter randomly across whole width of bar
	ax[0].scatter(x[i] + np.random.random(len(y_mcc[i])) * w - w / 2, y_mcc[i], color=edge_colors[i])

# repeat with F1 scores
(labels_fone,y_fone) = zip(*sorted(all_fone.items()))
ax[1].bar(x,
        height=[np.mean(yi) for yi in y_fone],
        yerr=[np.std(yi) for yi in y_fone],    # error bars
        capsize=12, # error bar cap width in points
        width=w,    # bar width
        tick_label=labels_fone,
        #color=(0,0,0,0,0,0,0,0),  # face color transparent
        color=('w','w','w','w','w','w','w','w'),  # face color transparent
        edgecolor=edge_colors,)

for i in range(len(x)):
        # distribute scatter randomly across whole width of bar
        ax[1].scatter(x[i] + np.random.random(len(y_fone[i])) * w - w / 2, y_fone[i], color=edge_colors[i])

ax[0].set_title("MCC scores\n29 shared drugs")
ax[0].tick_params(axis='x', labelrotation=90)
ax[1].set_title("F1 scores\n29 shared drugs")
ax[1].tick_params(axis='x', labelrotation=90)
plt.subplots_adjust(bottom=0.35)
plt.savefig(os.path.join(rdir,'dti_mcc_f1_29drugs.png'),format='png')	

