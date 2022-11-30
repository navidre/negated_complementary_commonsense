import argparse, sys
import pandas as pd
from tqdm import tqdm
sys.path.append('./')
from utils.gpt_3_utils import generate_zero_shot_using_gpt_3

if __name__ == "__main__":
    print('Loading spacy ...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv")
    parser.add_argument("--output", type=str, default="experiments/atomic_2020_eval/sampled_to_eval_negated_pred_with_gpt_3.tsv")
    parser.add_argument("--num_generations", type=int, default=3)
    args = parser.parse_args()

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
                generated_tail, response = generate_zero_shot_using_gpt_3(row_copy['prompt'], max_tokens=20)
                row_copy['generated_tail'] = generated_tail.replace('\n', ' ').strip()
                row_copy['full_text'] = f"{row_copy['prompt']} {row_copy['generated_tail']}".strip()
                # Append the row to generated_df
                generated_df = generated_df.append(row_copy, ignore_index=True)
        except:
            print('\nError in {}'.format(row['head']))
            import IPython; IPython. embed(); exit(1)

    # Save generated_df to args.output
    generated_df.to_csv(args.output, sep='\t', index=False)