import pandas as pd
import json
import argparse


def openai_check(prompt):
    pass

def get_inds_to_remove(df, coords):

    inds_to_remove = []
    for iter, (i, j, score) in enumerate(coords):
        if i in inds_to_remove or j in inds_to_remove:
            continue
        if score > 0.8:
            inds_to_remove.append(i)
            continue
        print(f'{iter+1}/{len(coords)}')
        print(df.iloc[i]['tasks'])
        print(df.iloc[j]['tasks'])
        print(f'Score: {score}')

        # match = input('Do these tasks match? (y/n): ')
        match = 'y'
        if match == 'y':
            inds_to_remove.append(i)

    return inds_to_remove
    



if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_file', type=str, default='')
    argparser.add_argument('--coord_file', type=str, default='')
    args = argparser.parse_args()

    if args.data_file == '' or args.coord_file == '':
        raise ValueError('Please provide a data file and a coordinate file')

    df = pd.read_csv(args.data_file)
    with open(args.coord_file, 'r') as f:
        coords = json.load(f)

    inds_to_remove = get_inds_to_remove(df, coords)

    prompt_list = []
    for i, row in df.iterrows():
        if i not in inds_to_remove:
            print(row['tasks'])
            prompt_list.append(row['tasks'])

    ok = input('\n\nDoes this look good? (y/n): ')
    if ok == 'y':
        with open(out_file:=args.data_file.replace('.csv', '') + '.json', 'w') as f:
            json.dump(prompt_list, f)

        print(f'\n\n*******************\nSaved to {out_file}')

    else:
        print('Exiting without saving')

    print(f'Complete!\n\n Final count: {len(prompt_list)} prompts')    