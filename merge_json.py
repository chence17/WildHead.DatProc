'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2024-01-26 16:44:04
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2024-01-26 16:44:07
FilePath: /DatProc/merge_json.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import json
import argparse

def merge_json(file1, file2, output_file):
    # Read data from the first JSON file
    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    # Read data from the second JSON file
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    # Merge the two JSON files
    merged_data = {**data1, **data2}

    # Write the merged data to the output file
    with open(output_file, 'w') as output_file:
        json.dump(merged_data, output_file, indent=2)

    print(f'Merged data written to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two JSON files.')
    parser.add_argument('file1', type=str, help='Path to the first JSON file')
    parser.add_argument('file2', type=str, help='Path to the second JSON file')
    parser.add_argument('output_file', type=str, help='Path to the output JSON file')

    args = parser.parse_args()

    merge_json(args.file1, args.file2, args.output_file)