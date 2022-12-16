import argparse, os, sys, json
import pandas as pd
from tqdm import tqdm
import numpy as np
import krippendorff
from statsmodels.stats import inter_rater as irr

CLASS_TO_INDEX = {
    'Makes sense': 1,
    'Sometimes makes sense': 2,
    'Does not make sense or Incorrect': 3,
    'First part and second part are not related! Or not enough information to judge.': 4,
    'Unfamiliar to me to judge': 5
}

import numpy as np

def calculate_fleiss_kappa(annotations):
    # Calculate the number of raters (columns in the annotations matrix)
    n_raters = annotations.shape[1]

    # Calculate the number of items (rows in the annotations matrix)
    n_items = annotations.shape[0]

    # Calculate the observed proportion of agreement for each item
    observed_proportion_agreement = np.sum(annotations == annotations[:,0][:,None], axis=1) / n_raters

    # Calculate the expected proportion of agreement
    expected_proportion_agreement = np.sum(annotations, axis=0) / (n_items * n_raters)

    # Calculate Fleiss' kappa
    kappa = (np.sum(observed_proportion_agreement) - np.sum(expected_proportion_agreement)) / (n_items - np.sum(expected_proportion_agreement))

    return kappa

def checkInput(rate, n):
    """ 
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError 
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    # assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer" 
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"

def fleissKappa(rate,n):
    """ 
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category 
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters   
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    checkInput(rate, n)

    #mean of the extent to which raters agree for the ith subject 
    PA = sum([(sum([i**2 for i in row])- n) / (n * (n - 1)) for row in rate])/N
    print("PA = ", PA)
    
    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j**2 for j in [sum([rows[i] for rows in rate])/(N*n) for i in range(k)]])
    print("PE =", PE)
    
    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)
    
    return kappa

def calculate_alpha_and_kappa_scores(annotations_df):
    """ Calculate alpha score (agreement between reviewers)
    """
    ratings = np.zeros((len(annotations_df), 3))
    for i, row in annotations_df.iterrows():
        for j in range(1, 4):
            # Correct, Incorrect, Unfamiliar
            review = int(float(row[f'review_{j}']))
            if review == 1 or review == 2:
                ratings[i, 0] += 1
            elif review == 3 or review == 4:
                ratings[i, 1] += 1
            else:
                ratings[i, 2] += 1
            # Five categories
            # ratings[i, int(float(row[f'review_{j}']))-1] += 1
    alpha = krippendorff.alpha(ratings)
    kappa = calculate_fleiss_kappa(ratings)
    irr_kappa = irr.fleiss_kappa(ratings, method='fleiss')
    import IPython; IPython. embed(); exit(1)
    return alpha, kappa

def update_out_tsv_from_manifest(mturk_path, out_tsv_path):
    """ Update the out_tsv file from prepare_generations_for_mturk_evaluation.py with the results from the manifest file
    """
    # Extracting folder name
    folder_name = [ch for ch in mturk_path.split('/') if ch != ''][-1]
    metadata_key = f'{folder_name}-metadata'
    # Get file name of out_tsv_path without extension
    out_filename = os.path.basename(out_tsv_path).split('.')[0]
    out_folder = os.path.dirname(out_tsv_path)
    # Loading output manifest as an array
    evaluations_list = []
    with open(f'{mturk_path}/manifests/output/output.manifest') as f:
        for line in f:
            line = json.loads(line)
            evaluations_list.append(line)
    # Loading output TSV file
    out_tsv_df = pd.read_csv(out_tsv_path, sep='\t', header=0)
    # Annotations path
    annotations_path = f'{mturk_path}/annotations/worker-response/iteration-1'
    # Iterate over the output manifest and update the output TSV file
    # Iterate over loaded TSV file
    manifest_index = 0
    for index, row in tqdm(out_tsv_df.iterrows()):
        # skip if already auto-evaluated
        if row['review'] != 0:
            continue
        # Extracting the source
        source = row['full_text']
        # Extracting the review
        consolidated_class_name = evaluations_list[manifest_index][metadata_key]['class-name']
        review = CLASS_TO_INDEX[consolidated_class_name]
        # Updating the output TSV file
        out_tsv_df.at[index, 'review'] = review
        # TODO: Add individual reviews
        # Load the individual reviews
        # worker_responses = evaluations[source][0][metadata_key]['worker-responses']
        annotation_json_folder_path = f'{annotations_path}/{manifest_index}'
        annotation_json_files = [f for f in os.listdir(annotation_json_folder_path) if os.path.isfile(os.path.join(annotation_json_folder_path, f))]
        # Iterate over the individual reviews
        assert len(annotation_json_files) == 1, f'Found more than one or no annotation files for {source}; Update implemetation to handle this case!'
        annotation_file_path = f'{annotation_json_folder_path}/{annotation_json_files[0]}'
        with open(annotation_file_path) as f:
            annotation_json = json.load(f)
            answer_jsons = annotation_json['answers']
            i = 1 # review index starts from 1
            for answer_json in answer_jsons:
                # Extracting the review
                class_name = answer_json['answerContent']['crowd-classifier']['label']
                review = CLASS_TO_INDEX[class_name]
                # Updating the output TSV file
                out_tsv_df.at[index, f'review_{i}'] = review
                i += 1
        # Update the manifest index (note that auto-evaluated rows are skipped in the output TSV file only)
        manifest_index += 1

    # Calculate alpha and kappa scores (agreement between reviewers) and store in a sepapte txt file
    alpha, kappa = calculate_alpha_and_kappa_scores(out_tsv_df)

    # Save the updated output TSV file
    out_tsv_df.to_csv(out_tsv_path, sep='\t', index=False)
    print(f'Updated {out_tsv_path} with the results from {mturk_path}.')

    # Save the information, including the alpha score, in a separate txt file
    out_filename_txt = f'{out_filename}.txt'
    with open(f'{out_folder}/{out_filename_txt}', 'w') as f:
        f.write(f'Krippendorf Alpha score: {alpha}\n')
        f.write(f'Fleiss Kappa score: {kappa}')
        print(f'Wrote the alpha and kappa scores to {out_folder}/{out_filename_txt}.')

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