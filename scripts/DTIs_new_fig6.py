# written 10-28-25 JLW
# to recreate figure 6
# for DTI revisions

import pickle,os,csv
import pandas as pd
from collections import defaultdict
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

rdir ='/Users/jenniferwilson/Documents/UCLA/drafts_papers_theses/Mohamad_DTIs/resubmission_2/' 
f = os.path.join(rdir,'SM 6- all_metrics_pathfx_dtis_v2.csv')
df = pd.read_csv(f)

dti_sources =['All_Targets', 'Baseline', 'ChEMBL', 'New_DrugBank', 'Pocket_Features', 'PubChem', 'STITCH', 'TTD']
dti_markers = [('blue','D'),('red','o'),('orange','+'),('black','*'),('cyan','X'),('green','s'),('magenta','^'),('gold','v')]

sdf = df[df.Drug.str.contains('donepezil')]

fig,ax = plt.subplots(1,2,figsize=(8,5),gridspec_kw={'width_ratios': [2, 1],})
for (ds,(sclr,mark_type)) in zip(dti_sources,dti_markers):
	ssdf = sdf[sdf.Drug.str.contains(ds)]
	dsens = ssdf['Sensitivity/Recall'].to_list()[0]
	dspec = ssdf['Specificity'].to_list()[0]
	ax[0].scatter(dspec,dsens,c=sclr,marker=mark_type)
	ax[1].plot([0,1],[dsens,dspec],linestyle='-',c=sclr,marker=mark_type,label=ds)

ax[0].set_title('Evaluation metrics\nfor donepezil')
ax[0].set_xlabel('specificity')
ax[0].set_ylabel('sensitivity')
ax[0].set_ylim([-.1,1.1])
ax[0].set_xlim([-.1,1.1])
ax[1].set_title('Sensitivity/Specificity changes\nfor donepezil')
ax[1].set_xticks([0,1])
ax[1].set_xticklabels(['sensitivity','specificity'], rotation=90) 
ax[1].legend(bbox_to_anchor=(2.3, 0.75), loc='upper right')
plt.subplots_adjust(bottom=0.3,right=0.75)
plt.savefig(os.path.join(rdir,'donepezil_all_metrics.png'),format='png')



