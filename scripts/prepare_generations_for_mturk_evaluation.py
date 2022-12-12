import argparse, os, sys
import pandas as pd
from tqdm import tqdm
sys.path.append('./')
from human_evaluate_generations import auto_evaluate_row

def generate_jsonl_for_mturk(in_file_path, out_jsonl_path):
    # read tsv file with columns head, relation, prompt, generated_tail, full_text
    df = pd.read_csv(in_file_path, sep='\t', header=0)
    # Opening output file for writing
    jsonl_file = open(out_jsonl_path, 'w')
    # Interate over rows and generate jsonl file
    for index, row in tqdm(df.iterrows(), desc="Generating jsonl file", total=len(df)):
        # skip if already evaluated
        if row['review_1'] != 0:
            continue
        # Add to jsonl file
        line_to_write = '{"source": "' + row['full_text'] + '"}\n' 
        jsonl_file.writelines([line_to_write])
    jsonl_file.close()
    print(f"Generated {out_jsonl_path}")

def auto_evaluate_generations(in_tsv, out_tsv):
    df = pd.read_csv(in_tsv, sep='\t', header=0)
    # Placeholder for review columns (three annotators)
    df['review_1'], df['review_2'], df['review_3'] = 0, 0, 0
    # Iterate over rows and auto-evaluate
    for index, row in df.iterrows():
        auto_evaluated, selection = auto_evaluate_row(index, row)
        if auto_evaluated:
            df.at[index, 'review_1'] = selection
            df.at[index, 'review_2'] = selection
            df.at[index, 'review_3'] = selection
    # Saving the dataframe
    df.to_csv(out_tsv, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tsv", type=str, default=f'./experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3.tsv')
    args = parser.parse_args()

    # Extracting file name
    filename = os.path.basename(args.in_tsv).split('.')[0]
    work_path = os.path.dirname(args.in_tsv)
    out_jsonl_path = os.path.join(work_path, f'{filename}_mturk.jsonl')
    out_tsv = os.path.join(work_path, f'{filename}_evaluated.tsv')

    # Auto-evaluating the generations with clear issues
    auto_evaluate_generations(args.in_tsv, out_tsv)

    # Generating the jsonl file with the ones not auto-evaluated
    generate_jsonl_for_mturk(out_tsv, out_jsonl_path)
    
    """ List of calls:
    - self_examples_eval:
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/self_samples_eval/few_shot_self_samples_to_eval_negated_preds_with_gpt_3.tsv
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/self_samples_eval/few_shot_self_samples_to_eval_preds_with_gpt_3.tsv
    - atomic_2020_eval:
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3.tsv
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/atomic_2020_eval/few_shot_sampled_to_eval_with_gpt_3.tsv
    """