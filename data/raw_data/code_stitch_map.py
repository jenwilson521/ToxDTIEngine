####################

import os
import csv

# Set the base directory where the files are located and create necessary directories
data_dir = '/.../stitch/filter/'
os.chdir(data_dir)
split_dir = os.path.join(data_dir, 'split_dir')
os.makedirs(split_dir, exist_ok=True)

# Check if split files already exist
def check_splits_exist(output_directory, expected_files=20):
    existing_files = len([name for name in os.listdir(output_directory) if os.path.isfile(os.path.join(output_directory, name))])
    return existing_files >= expected_files

# Split the large TSV file into smaller chunks if not already split
def split_large_file(large_tsv_file, output_directory, max_split_files=20):
    if check_splits_exist(output_directory, max_split_files):
        print("Split files already exist. Skipping the splitting process.")
        return
    large_tsv_path = os.path.join(data_dir, large_tsv_file)
    chunk_size = os.path.getsize(large_tsv_path) // max_split_files
    split_file_count = 0

    with open(large_tsv_path, 'r', encoding='utf-8') as input_file:
        for count, line in enumerate(input_file, start=1):
            if count % chunk_size == 1:
                if split_file_count > 0:  # Close the previous file if not the first
                    current_split_file.close()
                split_file_count += 1
                current_split_file = open(os.path.join(output_directory, f'split_{split_file_count}.tsv'), 'w', encoding='utf-8')
            current_split_file.write(line)
        if split_file_count > 0:  # Ensure the last file is closed
            current_split_file.close()
    print(f'Split into {split_file_count} smaller files.')

split_large_file('chemicals.v5.0.tsv', split_dir)

# Process each split file to filter and rename columns, and replace IDs in the stitch_base.tsv
def process_splits_and_replace_ids(input_directory, stitch_base_file, output_directory):
    stitch_base_path = os.path.join(data_dir, stitch_base_file)
    processed_files = []
    
    for i in range(1, 21):
        split_file_path = os.path.join(input_directory, f'split_{i}.tsv')
        output_file_path = os.path.join(output_directory, f'processed_split_{i}.tsv')
        processed_files.append(output_file_path)

        # Load the mapping from the split file
        mapping = {}
        with open(split_file_path, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)  # Skip header to read the file from the second line
            for row in reader:
                if len(row) >= 2:
                    chemical_id, chemical_name = row[0], row[1]
                    mapping[chemical_id] = chemical_name

        # Process stitch_base.tsv and replace IDs, only keep rows with mappings
        with open(stitch_base_path, mode='r', encoding='utf-8', newline='') as base_file, \
             open(output_file_path, mode='w', encoding='utf-8', newline='') as out_file:
            base_reader = csv.reader(base_file, delimiter='\t')
            writer = csv.writer(out_file, delimiter='\t')
            for row in base_reader:
                mapped_value = mapping.get(row[0])  # Get mapping if exists
                if mapped_value:  # If there's a mapping, write the row with the replaced value
                    row[0] = mapped_value
                    writer.writerow(row)

# Replace IDs in the stitch_base.tsv using the processed split files
process_splits_and_replace_ids(split_dir, 'stitch_base.tsv', data_dir)

# Combine all processed and ID-replaced files
def combine_processed_files(input_directory, output_file_name):
    output_file_path = os.path.join(data_dir, output_file_name)
    with open(output_file_path, mode='w', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        for i in range(1, 21):
            input_file_path = os.path.join(input_directory, f'processed_split_{i}.tsv')
            with open(input_file_path, mode='r', encoding='utf-8', newline='') as input_file:
                reader = csv.reader(input_file, delimiter='\t')
                for row in reader:
                    writer.writerow(row)

combine_processed_files(data_dir, 'final_combined_data.tsv')
print("All processes completed successfully.")

####################
