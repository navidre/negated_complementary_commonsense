
import os
import pandas as pd
from pandas.io.parsers import read_csv

def input_validator(value):
    try:
        value = int(value)
    except ValueError:
        return False
    return value >= 1 and value <= 5

def get_input(prompt, validator, on_validationerror):
    while True:
        value = input(prompt)
        if validator(value):
            return value
        print(on_validationerror)

def process_human_evaluation(work_path, in_tsv):
    # Extracting file name
    filename = os.path.basename(in_tsv).split('.')[0]
    out_filename = f'{filename}_self_evaluated'
    out_tsv = os.path.join(work_path, f'{out_filename}.tsv')
    # read tsv file with columns head, relation, prompt, generated_tail, full_text
    # if out_tsv exists, read it
    if os.path.exists(out_tsv):
        df = read_csv(out_tsv, sep='\t', header=0)
        # Review column already exists
    else:
        df = read_csv(in_tsv, sep='\t', header=0)
        df['review'] = 0
    
    # Tutorial
    os.system('clear')
    print('Tutorial: Please select a number as an answer from the following selection:')
    options_text = '1: Yes\n2: Sometimes\n3: No\n4: Invalid\n5: Unfamiliar to judge\n\n'
    print(options_text)
    input ('Press any key to continue to selections ...')
    os.system('clear')

    # Iterate over all rows in df
    correct_values = 0
    for index, row in df.iterrows():
        # skip if already reviewed
        if row['review'] > 0:
            continue
        # print row
        print(f'{row["full_text"]}\n')
        selection = get_input(f'Does this statement make sense? ', input_validator, 'Need to be an integer between 1 and 5!')
        if int(selection) <= 2:
            correct_values += 1
        df.loc[index, 'review'] = int(selection)
        # Saving every step to not lose the data
        df.to_csv(out_tsv, sep='\t', index=False)
        os.system('clear')
    # Saving the final result
    result = correct_values * 100.0 / len(df)
    print(f'Correct result percentage: {result}')
    with open(f'{work_path}/{out_filename}.txt', 'w') as f:
        f.write(f'Correct result percentage: {result}\n\n')
        f.write(options_text)


if __name__ == "__main__":
    work_path = './experiments/atomic_2020_eval'
    in_tsv = f'{work_path}/sampled_to_eval_negated_pred_with_gpt_3.tsv'
    process_human_evaluation(work_path, in_tsv)