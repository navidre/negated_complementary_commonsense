import argparse, os, sys, json
import pandas as pd
from tqdm import tqdm

CLASS_TO_INDEX = {
    'Makes sense': 1,
    'Sometimes makes sense': 2,
    'Does not make sense or Incorrect': 3,
    'First part and second part are not related! Or not enough information to judge.': 4,
    'Unfamiliar to me to judge': 5
}

def update_out_tsv_from_manifest(mturk_path, out_tsv_path):
    """ Update the out_tsv file from prepare_generations_for_mturk_evaluation.py with the results from the manifest file
    """
    # Extracting folder name
    folder_name = [ch for ch in mturk_path.split('/') if ch != ''][-1]
    metadata_key = f'{folder_name}-metadata'
    # Loading output manifest
    evaluations = {}
    with open(f'{mturk_path}/manifests/output/output.manifest') as f:
        # Loading output manifest as dict with source as key and answer content as value. Each key can have multiple values
        for line in f:
            line = json.loads(line)
            source = line['source']
            if source not in evaluations:
                evaluations[source] = []
            evaluations[source].append(line)
    # Loading output TSV file
    out_tsv_df = pd.read_csv(out_tsv_path, sep='\t', header=0)
    # Iterate over the output manifest and update the output TSV file
    # Iterate over loaded TSV file
    for index, row in tqdm(out_tsv_df.iterrows()):
        # skip if already auto-evaluated
        if row['review_1'] != 0:
            continue
        # Extracting the source
        source = row['full_text']
        # Extracting the review
        consolidated_class_name = evaluations[source][0][metadata_key]['class-name']
        review = CLASS_TO_INDEX[consolidated_class_name]
        # Updating the output TSV file
        out_tsv_df.at[index, 'review'] = review
        # Remove the evaluation from the dict
        evaluations[source].pop(0)
        # TODO: Add individual reviews
    # TODO: Sanity check and assert that all evaluations are used
    # TODO: Calculate alpha score (agreement between reviewers) and store in a sepapte txt file
    # Save the updated output TSV file
    out_tsv_df.to_csv(out_tsv_path, sep='\t', index=False)

if __name__ == "__main__":

    """ Sample AWS CLI command to download the results:
    aws s3 cp --recursive s3://negated-predicates/self_samples/ ./
    """

    """ Correspoding paths and output files:
    - self_examples_eval:
    Negated:
    --mturk_path ./experiments/self_samples_eval/mturk_results/negated-results/self-samples-negated/
    --out_tsv ./experiments/self_samples_eval/few_shot_self_samples_to_eval_negated_preds_with_gpt_3_evaluated.tsv
    Normal:
    --mturk_path ./experiments/self_samples_eval/mturk_results/normal-results/self-samples-negated/
    --out_tsv ./experiments/self_samples_eval/few_shot_self_samples_to_eval_with_gpt_3_evaluated.tsv
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mturk_path", type=str, default=f'./experiments/self_samples_eval/mturk_results/negated-results/self-samples-negated/')
    parser.add_argument("--out_tsv", type=str, default=f'./experiments/self_samples_eval/few_shot_self_samples_to_eval_negated_preds_with_gpt_3_evaluated.tsv')
    args = parser.parse_args()

    update_out_tsv_from_manifest(args.mturk_path, args.out_tsv)