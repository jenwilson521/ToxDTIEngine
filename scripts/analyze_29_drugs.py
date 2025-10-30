# written to calc AURC
# using Mohamad's results and mapping
# functions, written 10-30-25 JLW

import pickle,os,csv
from collections import defaultdict
import pandas as pd
from numpy import trapz
import numpy as np 
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

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

# first map AE string names to CUI terms
print('mapping AEs and CUI terms')
phene_ls = ['edema', 'gastric ulcer', 'neuroleptic malignant syndrome', 'delirium',
	'hyperlipidemia', 'completed suicide', 'hepatic necrosis',
	'tardive dyskinesia', 'proteinuria', 'hypertension', 'hemorrhage',
	'myocardial infarction', 'deep vein thrombosis', 'sepsis', 'cardiac arrest',
	'thrombocytopenia', 'agranulocytosis', 'stevens-johnson syndrome',
	'cerebral infarction', 'pancreatitis', 'peripheral neuropathy',
	'pulmonary edema', 'myopathy', 'pneumonia', 'anaphylaxis',
	'seizures', 'lung cyst', 'anemia', 'tachycardia',
	'prolonged qt interval', 'sleep disorders', 'sleep apnea syndromes']

p2c = pickle.load(open('../rscs/Pfx050120_all_phens_to_cuis.pkl','rb'))
name_ls = [name.lower() for name, value in p2c.items() if name.lower() in phene_ls]
AE_to_cuis = dict([(pname.lower(),value) for pname, value in p2c.items() if pname.lower() in phene_ls])
prim_cuis_to_AEs = dict([(v,k) for (k,v) in AE_to_cuis.items()]) # just for debugging

# next map primary AE CUI to acceptable matches by flipping map_to_orig_cui_dict
AE_cui_to_similar = defaultdict(set)
for (acc_cui,prim_cui) in map_to_orig_cui_dict.items():
	AE_cui_to_similar[prim_cui].add(acc_cui)

# next read AEs for each drug
print('reading AE data')
aef = '../data/Drugs_labeled_for_AEs.txt'
aedf = pd.read_csv(aef,delimiter='\t',dtype=str)
drugs_to_AEs = defaultdict(set)
for ae_name in aedf.columns:
	if type(ae_name)==type(1.0):
		continue
	for drug_name in aedf[ae_name].to_list():
		if type(drug_name)==type(1.0):
			continue
		drugs_to_AEs[drug_name.lower()].add(ae_name.lower())

# drugs for study, will have multiple sources 
print('loading PathFX_dti results')
rdir = '../results/pathfx_aim2_dti_run2/'
all_files = [x for x in os.walk(rdir)][1:]
dirs_files = [(sd,fname) for (sd,ssd,flist) in all_files for fname in flist if 'merged_neighborhood__assoc_table_.txt' in fname]
drugs_to_files = dict([(os.path.split(sd)[-1],os.path.join(sd,fname)) for (sd,fname) in dirs_files])

# test cases
#assf='../results/pathfx_aim2_dti_run2/acarbose_PubChem/acarbose_PubChem_merged_neighborhood__assoc_table_.txt'
#drug_AE_names = drugs_to_AEs['acarbose']; {'edema', 'thrombocytopenia'}
#pval_thr = 0.00055

def count_metrics(assf,drug_AE_names,pval_thr):
	asdf = pd.read_csv(assf,delimiter='\t')
	asdf['Benjamini-Hochberg'].astype('float')
	asdf_short = asdf[asdf['Benjamini-Hochberg']<pval_thr]
	pred_cuis = asdf_short.cui.to_list()
	# drug AEs matched to primary cuis and then acceptable matches
	drug_pos_primary = [AE_to_cuis[pn] for pn in drug_AE_names if pn in AE_to_cuis] # skip AEs like sleep disorder that don't match to CUIs
	tp = set([prim_cui for prim_cui in drug_pos_primary for acc_cui in AE_cui_to_similar[prim_cui] if prim_cui in pred_cuis or acc_cui in pred_cuis])
	fn = set(drug_pos_primary).difference(set(tp))
	tp_count = len(tp)
	fn_count = len(fn)
	# use remaining AEs as negatives
	drug_neg_primary = [AE_to_cuis[pn] for pn in AE_to_cuis.keys() if pn not in drug_AE_names]
	fp = set([prim_cui for prim_cui in drug_neg_primary for acc_cui in AE_cui_to_similar[prim_cui] if prim_cui in pred_cuis or acc_cui in pred_cuis])
	tn = set(AE_to_cuis.values()).difference(drug_pos_primary).difference(set(fp))
	fp_count = len(fp)
	tn_count = len(tn)
	return((tp_count,fn_count,fp_count,tn_count))
	

# loop through each source and count performance
rev_dir ='/Users/jenniferwilson/Documents/UCLA/drafts_papers_theses/Mohamad_DTIs/resubmission_2/'
affixes = [('New_DrugBank','black'), ('ChEMBL','orange'), ('PubChem','green'), ('STITCH','magenta'), ('TTD','gold'), ('Pocket_Features','cyan'), ('all','blue')]

# test cases
# dti_source = 'PubChem'
pval_array = np.logspace(-5,0,100)
fig,ax=plt.subplots()
# for dti_source in ['PubChem']:
for (dti_source,plt_clr) in affixes:
	print('counting for '+dti_source)
	all_drug_files = [(dn,f) for (dn,f) in drugs_to_files.items() if dti_source in dn]
	source_plot_data = []
	# for pval_thr in [0.0002,0.00055,0.1]:
	for pval_thr in pval_array:
		(all_tp,all_fn,all_fp,all_tn) = (0,0,0,0)
		for (dn,dnf) in all_drug_files:
			drug_name = dn.split('_')[0]
			drug_AEs = drugs_to_AEs[drug_name] 
			(tpc,fnc,fpc,tnc) = count_metrics(dnf,drug_AEs,pval_thr)
			all_tp+=tpc
			all_fn+=fnc
			all_fp+=fpc
			all_tn+=tnc
		s_t_sens = float(all_tp)/(all_tp + all_fn)
		s_t_spec = float(all_tn)/(all_fp + all_tn)
		source_plot_data.append((s_t_sens,(1-s_t_spec)))

	(x,y) = zip(*source_plot_data)
	pval_area = trapz(y,x)
	formatted_number = f"{pval_area:.3f}"
	ax.plot(x,y,c=plt_clr,label=dti_source+'\nAUROC:'+formatted_number)

ax.set_title('ROC curves all sources')
ax.legend(bbox_to_anchor=(1.2, 0.75))
ax.set_ylim([-.1,1.1])
ax.set_xlim([-.1,1.1])
plt.subplots_adjust(right=0.5)
# plt.savefig(os.path.join(rev_dir,dti_source+'_ROC_curve.png'),format='png')
plt.savefig(os.path.join(rev_dir,'ROC_curves_all.png'),format='png')
				
			
	
