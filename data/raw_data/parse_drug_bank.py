# Written to get itneraction data
# for drugs and their targets
# copied from https://github.com/dhimmel/drugbank/blob/gh-pages/parse.ipynb
#  4-20-17 JLW

import os,csv,gzip,collections,re,io,json,sys,requests,pandas
import xml.etree.ElementTree as ET

print("sys.version")
print(sys.version)

#tree = ET.parse('../data/full_database.xml')
###
tree = ET.parse('full_database.xml')
###
root = tree.getroot()
#[(child.tag,child.attrib) for child in root]
#drugs = [drug for drug in root.findall('{http://www.drugbank.ca}drug')]
#dnames = [(drug,drug.find('name').text) for drug in drugs]
#drug_names = [root[i][3].text for i in range(0,len(root))]

ns = '{http://www.drugbank.ca}'
inchikey_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChIKey']/{ns}value"
inchi_template = "{ns}calculated-properties/{ns}property[{ns}kind='InChI']/{ns}value"

print('gathering drugs, codes, and stuff')
sys.stdout.flush()
rows = list()
for i, drug in enumerate(root):
    row = collections.OrderedDict()
    assert drug.tag == ns + 'drug'
    row['type'] = drug.get('type')
    row['drugbank_id'] = drug.findtext(ns + "drugbank-id[@primary='true']")
    row['name'] = drug.findtext(ns + "name")
    row['description'] = drug.findtext(ns + "description")
    row['groups'] = [group.text for group in
        drug.findall("{ns}groups/{ns}group".format(ns = ns))]
    row['atc_codes'] = [code.get('code') for code in
        drug.findall("{ns}atc-codes/{ns}atc-code".format(ns = ns))]
    row['categories'] = [x.findtext(ns + 'category') for x in
        drug.findall("{ns}categories/{ns}category".format(ns = ns))]
    row['inchi'] = drug.findtext(inchi_template.format(ns = ns))
    row['inchikey'] = drug.findtext(inchikey_template.format(ns = ns))
    
    # Add drug aliases
    aliases = {
        elem.text for elem in 
        drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
        drug.findall("{ns}synonyms/{ns}synonym[@language='English']".format(ns = ns)) +
        drug.findall("{ns}international-brands/{ns}international-brand".format(ns = ns)) +
        drug.findall("{ns}products/{ns}product/{ns}name".format(ns = ns))

    }
    aliases.add(row['name'])
    row['aliases'] = sorted(aliases)

    rows.append(row)

def collapse_list_values(row):
    for key, value in row.items():
        if isinstance(value, list):
            row[key] = '|'.join(value)
    return row

rows = list(map(collapse_list_values, rows))

columns = ['drugbank_id', 'name', 'type', 'groups', 'atc_codes', 'categories', 'inchikey', 'inchi', 'description']
drugbank_df = pandas.DataFrame.from_dict(rows)[columns]

print('save drug info to tsv')
sys.stdout.flush()
#path = os.path.join('..', 'data', 'drugbank.tsv')
#drugbank_df.to_csv(path, sep='\t', index=False,encoding='utf-8')
###
drugbank_df.to_csv('drugbank.tsv', sep='\t', index=False,encoding='utf-8')
###
print('gathering drug actions')
sys.stdout.flush()
protein_rows = list()
for i, drug in enumerate(root):
    drugbank_id = drug.findtext(ns + "drugbank-id[@primary='true']")
    for category in ['target', 'enzyme', 'carrier', 'transporter']:
        proteins = drug.findall('{ns}{cat}s/{ns}{cat}'.format(ns=ns, cat=category))
        for protein in proteins:
            row = {'drugbank_id': drugbank_id, 'category': category}
            row['organism'] = protein.findtext('{}organism'.format(ns))
            row['known_action'] = protein.findtext('{}known-action'.format(ns))
            actions = protein.findall('{ns}actions/{ns}action'.format(ns=ns))
            row['actions'] = '|'.join(action.text for action in actions)
            uniprot_ids = [polypep.text for polypep in protein.findall(
                "{ns}polypeptide/{ns}external-identifiers/{ns}external-identifier[{ns}resource='UniProtKB']/{ns}identifier".format(ns=ns))]            
            if len(uniprot_ids) != 1:
                continue
            row['uniprot_id'] = uniprot_ids[0]
#            ref_text = protein.findtext("{ns}references[@format='textile']".format(ns=ns))
#            print(type(ref_text))
#            pmids = re.findall(r'pubmed/([0-9]+)', ref_text)
#            row['pubmed_ids'] = '|'.join(pmids)
            protein_rows.append(row)

protein_df = pandas.DataFrame.from_dict(protein_rows)

# Read our uniprot to entrez_gene mapping
response = requests.get('http://git.dhimmel.com/uniprot/data/map/GeneID.tsv.gz', stream=True)
text = io.TextIOWrapper(gzip.GzipFile(fileobj=response.raw))
uniprot_df = pandas.read_table(text, engine='python')
uniprot_df.rename(columns={'uniprot': 'uniprot_id', 'GeneID': 'entrez_gene_id'}, inplace=True)

# merge uniprot mapping with protein_df
entrez_df = protein_df.merge(uniprot_df, how='inner')

columns = ['drugbank_id', 'category', 'uniprot_id', 'entrez_gene_id', 'organism',
           #'known_action', 'actions', 'pubmed_ids']
           'known_action', 'actions']
entrez_df = entrez_df[columns]

print('saving drug action to output')
#path = os.path.join('..', 'data', 'proteins.tsv')
#entrez_df.to_csv(path, sep='\t', index=False)
###
entrez_df.to_csv('proteins.tsv', sep='\t', index=False)
###
