####################

# read the list of cids from the file
cid_list=()
while IFS=':' read -r line; do
  cid_list+=("$line")
done < "cid_list.txt"

# download data for each cid
for num in "${cid_list[@]}"; do
  wget -q "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query={\"download\":\"*\",\"collection\":\"consolidatedcompoundtarget\",\"where\":{\"ands\":[{\"cid\":\"$num\"}]},\"start\":1,\"limit\":10000000}" -O "${num}.csv"
  echo "Downloaded data for CID ${num}."
done

####################
