import argparse, os
import pandas as pd
from tqdm import tqdm

def generate_jsonl_for_mturk(args):
    # Extracting file name
    filename = os.path.basename(args.in_tsv).split('.')[0]
    out_filename = f'{filename}_mturk'
    work_path = os.path.dirname(args.in_tsv)
    out_jsonl_path = os.path.join(work_path, f'{out_filename}.jsonl')
    # read tsv file with columns head, relation, prompt, generated_tail, full_text
    df = pd.read_csv(args.in_tsv, sep='\t', header=0)
    # Opening output file for writing
    jsonl_file = open(out_jsonl_path, 'w')
    # Interate over rows and generate jsonl file
    for index, row in tqdm(df.iterrows(), desc="Generating jsonl file", total=len(df)):
        line_to_write = '{"source": "' + row['full_text'] + '"}\n' 
        jsonl_file.writelines([line_to_write])
    jsonl_file.close()
    print(f"Generated {out_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tsv", type=str, default=f'./experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3.tsv')
    args = parser.parse_args()
    generate_jsonl_for_mturk(args)

    """ Sample run:
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/self_samples_eval/few_shot_self_samples_to_eval_negated_preds_with_gpt_3.tsv
    """