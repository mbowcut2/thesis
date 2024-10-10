import argparse
import os
import json


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, default='')


    args = parser.parse_args()

    if args.target_folder == '':
        raise ValueError('Please provide a target folder')
    
    all_data = []

    print(f'Merging datasets from {args.target_folder}: {os.listdir(args.target_folder)}')
    
    for filename in os.listdir(args.target_folder):
        if filename.endswith('.json'):
            with open(os.path.join(args.target_folder, filename), 'r') as f:
                try:
                    data = json.load(f).get('tasks', [])
                except:
                    print(f'Error loading {filename}')
                    continue
                all_data.extend(data)

    with open(os.path.join(args.target_folder, f'merged.json'), 'w') as f:
        json.dump(all_data, f)

    print(f'Merged {len(all_data)} tasks from {len(os.listdir(args.target_folder))} files')