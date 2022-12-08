import argparse, sys, os
import pandas as pd
from tqdm import tqdm
sys.path.append('./')
from utils.gpt_3_utils import generate_zero_shot_using_gpt_3, NEGATED_FEW_SHOT_PROMPT, generate_few_shot_using_gpt_3, FEW_SHOT_PROMPT

if __name__ == "__main__":
    print('Loading spacy ...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv")
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--style", type=str, default="few_shot", choices=["zero_shot", "few_shot"])
    parser.add_argument("--limited_preds", action="store_true")
    parser.add_argument("--negated", action="store_true")
    args = parser.parse_args()

    #TODO: Make sure we do not overwrite the existing generations (if generated file exists) 
    #TODO: and we do not re-do generation for the same prompt (if the prompt is already in the generated file)

    """Sample calls:
    - For normal (non-negated), few-shot generation with limited_preds:
    python scripts/generate_objects_using_gpt_3.py --input experiments/atomic_2020_eval/sampled_to_eval.tsv --style few_shot --limited_preds
    - For negated, few-shot generation without limited_preds:
    python scripts/generate_objects_using_gpt_3.py --input experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv --style few_shot --negated
    """

    if args.limited_preds:
        # These limited preds are chosen as they resulted in the lowest accuracy in negated scenarios
        limited_preds = ['xWant', 'xReact', 'oWant', 'CapableOf', 'Desires', 'HinderedBy', 'isBefore', 'isAfter', 'AtLocation', 'HasSubevent', 'ObjectUse']

    # Extracting file name and defining the out file name and path
    filename = os.path.basename(args.input).split('.')[0]
    out_filename = f'{args.style}_{filename}_with_gpt_3'
    work_path = os.path.dirname(args.input)
    out_tsv = os.path.join(work_path, f'{out_filename}.tsv')
    # Load args.input as a pandas dataframe and ignore the first row
    df = pd.read_csv(args.input, sep="\t", header=0)
    # Remove 'tail' column
    df = df.drop(columns=['tail'])
    # Placeholder dataframe to store generated rows, which has args.num_generations rows for each row in df
    generated_df = pd.DataFrame(columns=['head', 'relation', 'prompt', 'generated_tail', 'full_text'])

    # Iterate over all rows in df
    progress_bar = tqdm(df.iterrows())
    for index, row in progress_bar:
        progress_bar.set_description("Processing {}-{}".format(row['head'], row['relation']))
        # Skip if the relation is not in limited_preds
        if args.limited_preds and row['relation'] not in limited_preds:
            continue
        # Exception handling if there is an error in the following lines
        try:
            # Make a copy of row and store in row_copy
            row_copy = row.copy()
            # Add 'generated_tail' and 'full_text' columns to row_copy
            row_copy['generated_tail'] = ''
            row_copy['full_text'] = ''
            # Generate args.num_generations rows for each row in df
            for i in range(args.num_generations):
                # Generate zero shot using GPT-3
                if args.style == "zero_shot":
                    generated_tail, response = generate_zero_shot_using_gpt_3(row_copy['prompt'], max_tokens=20)
                elif args.style == "few_shot":
                    if args.negated:
                        generated_tail, response = generate_few_shot_using_gpt_3(NEGATED_FEW_SHOT_PROMPT, row_copy['prompt'], max_tokens=20)
                    else:
                        generated_tail, response = generate_few_shot_using_gpt_3(FEW_SHOT_PROMPT, row_copy['prompt'], max_tokens=20)
                else:
                    raise NotImplementedError
                row_copy['generated_tail'] = generated_tail.replace('\n', ' ').strip()
                row_copy['full_text'] = f"{row_copy['prompt']} {row_copy['generated_tail']}".strip()
                # Append the row to generated_df
                generated_df = generated_df.append(row_copy, ignore_index=True)
        except:
            print('\nError in {}'.format(row['head']))
            import IPython; IPython. embed(); exit(1)

        # Save generated_df as a tsv file in every step
        generated_df.to_csv(out_tsv, sep='\t', index=False)