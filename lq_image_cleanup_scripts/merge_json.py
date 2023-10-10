'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-10 21:36:04
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-10 21:42:31
FilePath: /DatProc/lq_image_cleanup_scripts/merge_json.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json
import argparse
from tqdm import tqdm

def config_parser():
    parser = argparse.ArgumentParser(description='This script removes the wrong campose images from the dataset.json file.')
    parser.add_argument('--src_dir', type=str, default='/data1/chence/PanoHeadData/single_view/dataset.json', help='The source json file.')
    return parser.parse_args()

args = config_parser()
# Directory containing the JSON files
input_directory = args.src_dir

# List to store the data from individual JSON files
data = []

# Iterate over each JSON file in the input directory
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith("-12.json"):
        with open(os.path.join(input_directory, filename), "r") as file:
            json_data = json.load(file)
            data.append(json_data)

# Merge the data into a single dictionary
merged_data = {}
for item in data:
    merged_data.update(item)

# Output file path for the merged JSON
output_file_path = os.path.join(input_directory, "meta.json")

# Write the merged JSON data to the output file
with open(output_file_path, "w") as output:
    json.dump(merged_data, output, indent=4)

print(f"Merged JSON data saved to {output_file_path}")