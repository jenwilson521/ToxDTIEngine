### PathFX version created: Pfxdti

# IMPORT STATEMENTS
import pickle,os
from optparse import OptionParser
from collections import defaultdict
from scipy.stats import hypergeom
global umls_rank
umls_rank = 50
# import association file data
rscs_dir = '../rscs/'

# DATA UPDATE-SPECIFIC VARIABLES
phen_to_cui = pickle.load(open('../rscs/Pfxdti_all_phens_to_cuis.pkl','rb'))
cui_to_phens = pickle.load(open('../rscs/Pfxdti_cui_to_phens.pkl','rb'))
cui_to_genes = pickle.load(open('../rscs/Pfxdti_merged_unique_cuis2genes.pkl','rb'))
intome_size = pickle.load(open('../rscs/pfx041520_intome_size.pkl','rb'))
genes_to_cuis = pickle.load(open('../rscs/Pfxdti_merged_genes_to_cuis.pkl','rb'))
sourced_phens = pickle.load(open('../rscs/Pfxdti_sourced_phens.pkl','rb'))
expect_pvals = pickle.load(open('../results/Pfxdtirandom_networks/Pfxdti_expected_pvalue_summary.pkl','rb'))


# BASE METHODS
def get_network_interactions(f):
	fdata = [l.strip().split('\t') for l in open(f,'r').readlines()]
	return fdata

def get_node_list(f):
	fdata = [l.strip().split('\t') for l in open(f,'r').readlines()]
	sourcen = [l[0] for l in fdata]
	sinkn = [l[1] for l in fdata]
	node_list = list(set(sourcen+sinkn))
	if '' in node_list:
		node_list.remove('')
	return node_list
	
def get_assoc(node_list):
	# pull assocationes
	assoc_count = defaultdict(int)
	assoc_genes = defaultdict(list)
	for n in node_list:
		if n in genes_to_cuis:
			for cui in genes_to_cuis[n]:
				assoc_count[cui]+=1
				assoc_genes[cui].append(n)

	return (assoc_count,assoc_genes)


def calc_hyp(node_list,cui_to_genes,N,Q):
	n = len(node_list)
	(assoc_count,assoc_genes) = get_assoc(node_list)

	assoc_analy = []
	for (a,k) in assoc_count.items():
		K = len(cui_to_genes[a])
		prb = 1 - hypergeom.cdf(k,N,K,n)	
		assoc_analy.append([a,k,K,prb])
	# Q = 0.001
	sort_assoc = sorted(assoc_analy,key = lambda x:(x[3],x[0]))
	m = len(sort_assoc)
	mhc_assoc = []
	for (i,[a,k,K,prb]) in enumerate(sort_assoc):
		BH = (float(i+1)/m)*Q # calculate Benjamini-Hochberg based on ranked data
		mhc_assoc.append([i+1,a,k,K,prb,BH])
	sig_assoc = []
	for [rank,phen,assnet,assint,prb,BH] in mhc_assoc:
		if prb<BH and assint >24:
			genes = sorted(assoc_genes[phen])
			gene_str = ','.join(genes)
			phen_term = cui_to_phens[phen][0] # use the first phenotype as the descriptor
			sig_assoc.append([rank,phen_term,phen,assnet,assint,prb,BH,gene_str])
		elif prb>BH:
			break
	return sig_assoc

def background_check(num_t,sig_assoc):
	bgd_chckd = []
	for [rank,phen_term,phen,assnet,assint,prb,BH,gene_str] in sig_assoc:
		if phen in expect_pvals[num_t]:
			exp_pv = expect_pvals[num_t][phen]
		else:
			exp_pv = 1.0
		if BH <	exp_pv:
			bgd_chckd.append([rank,phen_term,phen,assnet,assint,prb,BH,gene_str])
	return bgd_chckd

def write_to_output(sig_assoc,outfname):
	outf = open(outfname,'w')
	outf.write('\t'.join(['rank','phenotype','cui','assoc in neigh','assoc in intom','probability','Benjamini-Hochberg','genes','\n']))
	if len(sig_assoc) >0:
		for line in sig_assoc:
			outf.write('\t'.join([str(x) for x in line])+'\n')
	outf.close()

def write_sources(sig_assoc,outfname):
	outf = open(outfname,'w')
	hed = ['Gene','CUI','Source Databases','\n']
	outf.write('\t'.join(hed))
	for [rank,phen_term,phen,assnet,assint,prb,BH,gene_str] in sig_assoc:
		for g in gene_str.split(','):
			cgkey = (g,phen)
			db_source = sorted(sourced_phens[cgkey])
			db_s_str = ','.join(db_source)
			outline = '\t'.join([g,phen,phen_term,db_s_str,'\n'])
			outf.write(outline)
	outf.close()

def write_cui_list(sig_assoc,outfname):
	if len(sig_assoc) >=100:
		cui_list = [x[2] for x in sig_assoc][:umls_rank]
	else:
		cui_list = [x[2] for x in sig_assoc]
#	pickle.dump(cui_list,open(outfname,'wb'))
	outf = open(outfname,'w')
	outf.write('\n'.join(cui_list))
	outf.close()
	
def main():
	parser=OptionParser()

	parser.add_option('-f','--file',dest='netf',help='Tab-separated network file of HUGO gene symbols')
	parser.add_option('-a','--analysis_name',dest='aname',help='Name of analysis, will be appended to output files; experiment date is suggested')
	parser.add_option('-d','--dir',dest='res_dir',help='Results directory. If none provided, a directory will be created matching the analysis name in the ../results/ dir')
	parser.add_option('-n','--numt',dest='numt',help='number of targets used as input')

	(options,args) = parser.parse_args()

	# check if results directory exists, otherwise assign based on analysis
	if options.res_dir is None:
		rdir = '../results/'+options.aname+'/'
	else:
		rdir = options.res_dir

	print('gathering network data')
	# Gather network data
	node_list = get_node_list(os.path.join(rdir,options.netf))	
	net_int = get_network_interactions(os.path.join(rdir,options.netf))
	
	print('calculating hypergeometric probabilities')
	net_assoc = get_assoc(node_list)

	N = intome_size
	Q = 0.001
	# Multiple hypothesis correct
	sig_assoc = calc_hyp(node_list,cui_to_genes,N,Q)
	# compare to expected p-values
	background_checked = background_check(options.numt,sig_assoc)

	print('saving to output')
	froot = os.path.splitext(options.netf)[0]
	outfname = os.path.join(rdir,'_'.join([froot,'assoc','table','.txt']))
	write_to_output(sig_assoc,outfname)
	outfname = os.path.join(rdir,'_'.join([froot,'assoc','database','sources','.txt']))
	write_sources(sig_assoc,outfname)
	outfname = os.path.join(rdir,'_'.join([froot,'cui','list','.txt']))
	write_cui_list(sig_assoc,outfname)

if __name__ == "__main__":
	main()
