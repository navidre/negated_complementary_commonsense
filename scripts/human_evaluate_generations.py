
import os
import pandas as pd
from pandas.io.parsers import read_csv

def input_validator(value):
    try:
        value = int(value)
    except ValueError:
        return False
    return value >= 1 and value <= 4

def get_input(prompt, validator, on_validationerror):
    while True:
        value = input(prompt)
        if validator(value):
            return value
        print(on_validationerror)

def auto_evaluate_row(index, row):
    """ Selection 3 (invalid) if the generated tail is empty or contains ___

    Args:
        index (_type_): index of row
        row (_type_): row to be evaluated

    Returns:
        auto_evaluated, selection: True if auto evaluated, False otherwise; The selection made
    """
    if type(row['generated_tail']) == float:
        return True, 3
    
    if row['generated_tail'] == '':
        return True, 3

    if '___' in row['generated_tail']:
        return True, 3

    # All other cases    
    return False, 0

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
    options_text = '1: Makes sense\n2: Sometimes\n3: Incorrect or does not make sense\n4: Unfamiliar to judge\n\n'
    print(options_text)
    input ('Press any key to continue to selections ...')
    os.system('clear')

    # Iterate over all rows in df
    correct_values = 0
    for index, row in df.iterrows():
        # skip if already reviewed
        if row['review'] > 0:
            continue
        # Print progress
        print(f'Row {index+1} out of {len(df)} rows:\n')
        # Auto evaluate if specific issues happen
        auto_evaluated, selection = auto_evaluate_row(index, row)
        # Manually evaluate
        if not auto_evaluated:
            print(f'{row["full_text"]}\n')
            print(options_text)
            selection = get_input(f'Does this statement make sense? ', input_validator, 'Need to be an integer between 1 and 4!')
            if int(selection) <= 2:
                correct_values += 1
        # Adding the selection to the dataframe, either manually or automatically   
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