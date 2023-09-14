import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run Frontal View Pipeline')
    parser.add_argument('-f', '--file', type=str, help='path to json metadata', default='/home/shitianhao/project/DatProc/assets/mh_dataset')
    parser.add_argument('-o', '--output_dir', type=str, help='path to output dir', default='/home/shitianhao/project/DatProc/utils/stats')
    parser.add_argument('-j', '--num_processes', type=int, help='number of processes', default=32)
    args, _ = parser.parse_known_args()
    return args

def main(args):
    with open(args.file, 'r') as f:
        data = json.load(f)
    dataset_path = data.keys()[0]
    print(dataset_path)