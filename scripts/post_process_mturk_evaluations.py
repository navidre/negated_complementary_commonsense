import argparse, os, sys, json
import pandas as pd
from tqdm import tqdm
import numpy as np
import krippendorff
from statsmodels.stats import inter_rater as irr
import numpy as np
from statistics import mode, StatisticsError

# Class class to index mapping
CLASS_TO_INDEX = {
    'Makes sense': 1,
    'Sometimes makes sense': 2,
    'Does not make sense or Incorrect': 3,
    'First part and second part are not related! Or not enough information to judge': 4,
    'Unfamiliar to me to judge': 5
}

# Experimental class to index mapping
# CLASS_TO_INDEX = {
#     'Correct': 1,
#     'Sometimes correct': 2,
#     'Incorrect': 3,
#     'Not enough information to judge': 4,
#     'Unfamiliar to me to judge': 5
# }

def calculate_alpha_and_kappa_scores(annotations_df):
    """ Calculate alpha score (agreement between reviewers)
    """
    two_category_ratings = np.zeros((len(annotations_df), 2))
    correct_incorrect_ratings = np.zeros((len(annotations_df), 2))
    three_category_ratings = np.zeros((len(annotations_df), 3))
    five_category_ratings = np.zeros((len(annotations_df), 5))
    for i, row in annotations_df.iterrows():
        for j in range(1, 4):
            # Correct, Incorrect, Unfamiliar
            review = int(float(row[f'review_{j}']))
            # Three categories
            if review == 1 or review == 2:
                three_category_ratings[i, 0] += 1
            elif review == 3 or review == 4:
                three_category_ratings[i, 1] += 1
            else:
                three_category_ratings[i, 2] += 1
            # Two categories
            if review == 1 or review == 2:
                two_category_ratings[i, 0] += 1
            else:
                two_category_ratings[i, 1] += 1
            # Correct/Incorrect categories
            if review == 1 or review == 2:
                correct_incorrect_ratings[i, 0] += 1
            elif review == 3 or review == 4:
                correct_incorrect_ratings[i, 1] += 1
            # Five categories
            five_category_ratings[i, int(float(row[f'review_{j}']))-1] += 1
    two_category_alpha = krippendorff.alpha(two_category_ratings)
    correct_incorrect_alpha = krippendorff.alpha(correct_incorrect_ratings)
    three_category_alpha = krippendorff.alpha(three_category_ratings)
    five_category_alpha = krippendorff.alpha(five_category_ratings)
    return two_category_alpha, correct_incorrect_alpha, three_category_alpha, five_category_alpha

def majority(votes, aws_vote):
    majority_vote, absolute_majority_vote = 0, 0
    # Check if there is a duplicate to result in majority vote
    if len(set(votes)) != len(votes):
        majority_vote = mode(votes)
        absolute_majority_vote = majority_vote
    else:  
        # If there is no duplicate, we categorize the votes. 1 & 2 are positive, 3 & 4 are negative, 5 is neutral
        positive_votes = [vote for vote in votes if vote == 1 or vote == 2]
        negative_votes = [vote for vote in votes if vote == 3 or vote == 4]
        neutral_votes = [vote for vote in votes if vote == 5]
        # If there are more positive votes than negative votes, we assign 1 as majority vote
        if len(positive_votes) > len(negative_votes):
            majority_vote = 1
            absolute_majority_vote = 1
        # If there are more negative votes than positive votes, we assign 3 as majority vote
        elif len(negative_votes) > len(positive_votes):
            majority_vote = 3
            absolute_majority_vote = 3
        # If there are more neutral votes than positive or negative votes, we assign 5 as majority vote
        elif len(neutral_votes) > len(positive_votes) and len(neutral_votes) > len(negative_votes):
            majority_vote = 5
            absolute_majority_vote = 5
        # Else, we assign the AWS vote as majority vote
        else:
            majority_vote = aws_vote
            absolute_majority_vote = 0
    return majority_vote, absolute_majority_vote


def update_out_tsv_from_manifest(mturk_path, out_tsv_path):
    """ Update the out_tsv file from prepare_generations_for_mturk_evaluation.py with the results from the manifest file
    """
    # Get experiments path
    experiments_path = '/'.join(mturk_path.split('/')[:-2])
    # Extracting folder name
    folder_name = [ch for ch in mturk_path.split('/') if ch != ''][-1]
    # New folder with annotations folder name under experiments path
    annotations_folder = f'{experiments_path}/{folder_name}'
    # Create annotations folder if it does not exist
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
    # Copy out_tsv_path file to annotations folder
    os.system(f'cp {out_tsv_path} {annotations_folder}')
    # Updated out_tsv_path
    out_tsv_path = f'{annotations_folder}/{os.path.basename(out_tsv_path)}'
    # Get metadata key
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
    # Number of folders under annotations path
    num_of_annotations = len(os.listdir(annotations_path))
    # Iterate over the output manifest and update the output TSV file
    # Iterate over loaded TSV file
    manifest_index = 0
    for index, row in tqdm(out_tsv_df.iterrows()):
        # skip if already auto-evaluated
        if row['review'] != 0 and (row['review_1'] == row['review_2'] == row['review_3'] == row['review']) and row['flagged_answer']:
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
    
    # Assert that all annotations are processed
    assert manifest_index == num_of_annotations, f'Number of annotations ({num_of_annotations}) does not match the number of processed annotations ({manifest_index})'

    # Calculating majority vote
    # Add empty columns for majority vote
    # majority_vote considers Sagemaker's vote if there is no majority
    # absolute_majority_vote sets 0 if there is no majority
    out_tsv_df['majority_vote'] = 0
    out_tsv_df['absolute_majority_vote'] = 0
    for index, row in out_tsv_df.iterrows():
        votes = [int(row['review_1']), int(row['review_2']), int(row['review_3'])]
        aws_vote = int(row['review'])
        majority_vote, absolute_majority_vote = majority(votes, aws_vote)
        out_tsv_df.at[index, 'majority_vote'] = majority_vote
        out_tsv_df.at[index, 'absolute_majority_vote'] = absolute_majority_vote

    # Calculate alpha score (agreement between reviewers) and store in a sepapte txt file
    two_category_alpha, correct_incorrect_alpha, three_category_alpha, five_category_alpha = calculate_alpha_and_kappa_scores(out_tsv_df)

    # Save the updated output TSV file
    out_tsv_df.to_csv(out_tsv_path, sep='\t', index=False)
    print(f'Updated {out_tsv_path} with the results from {mturk_path}.')

    # Save the information, including the alpha score, in a separate txt file
    out_filename_txt = f'{out_filename}.txt'
    with open(f'{out_folder}/{out_filename_txt}', 'w') as f:
        f.write(f'Two-Categorry Krippendorf Alpha score: {two_category_alpha}\n')
        f.write(f'Correct/Incorrect Krippendorf Alpha score: {correct_incorrect_alpha}\n')
        f.write(f'Three-Categorry Krippendorf Alpha score: {three_category_alpha}\n')
        f.write(f'Five-Categorry Krippendorf Alpha score: {five_category_alpha}\n')
        print(f'Wrote the alpha scores to {out_folder}/{out_filename_txt}.')

if __name__ == "__main__":

    """ Sample AWS CLI command to download the results. We need to go to mturk results folder and run this command:
    aws s3 cp --recursive s3://negated-predicates/self_samples/ ./
    aws s3 cp --recursive s3://[folder-path-to-results] ./
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