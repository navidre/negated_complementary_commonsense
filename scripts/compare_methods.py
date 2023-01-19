import argparse, os, json
from tqdm import tqdm
import pandas as pd
import krippendorff
import numpy as np

EXPERIMENTS = { 
    # 'atmoic2020_limited_preds': {
    #     'few_shot': 'experiments/sampled_10_atomic2020_few_shot_limited_preds',
    #     'few_shot_qa': 'experiments/sampled_10_atomic2020_few_shot_qa_limited_preds',
    #     'cot_qa': 'experiments/sampled_10_atomic2020_cot_qa_limited_preds',
    #     'updated_cot_qa': 'experiments/sampled_10_atomic2020_updated_cot_qa_limited_preds',
    #     # 'cot_qa_neg_teach': 'experiments/sampled_10_atomic2020_cot_qa_neg_teach_limited_preds',
    #     'cot_qa_neg_teach_var_temp': 'experiments/sampled_10_atomic2020_cot_qa_neg_teach_var_temp_limited_preds',
    #     'cot_qa_updated_neg_teach_var_temp': 'experiments/sampled_10_atomic2020_cot_qa_updated_neg_teach_var_temp_limited_preds',
    # },
    'self_examples': {
        'few_shot': '',
        'few_shot_qa': '',
        'cot_qa': ''
    },
    'atomic2020_ten_preds': {
        'few_shot_qa': 'experiments/sampled_10_atomic2020_few_shot_qa_limited_preds_ten_atomic_preds',
        'cot_qa_updated_neg_teach_var_temp': 'experiments/sampled_10_atomic2020_cot_qa_updated_neg_teach_var_temp_limited_preds_ten_atomic_preds'
    }
}

def extract_accurcy_from_json(json_filepath):
    accuracy = 0
    # check if file exists and extract the accuracy
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as f:
            results = json.load(f)
            accuracy = int(results['total']['correct'])*100.0/int(results['total']['count'])
    return accuracy

def add_new_vals_to_merged_evals(merged_evaluation_df, eval_df, review_column_count):
    if len(merged_evaluation_df) == 0:
        # Drop columns review, majority_vote, and absolute_majority_vote from eval_df
        eval_df.drop(columns=['review', 'majority_vote', 'absolute_majority_vote'], inplace=True)
        # Assign the current evaluation to merged_negated_evaluation
        merged_evaluation_df = eval_df
    else:
        # Select columns review_1, review_2, and review_3 of the current evaluation
        current_evals = eval_df[['review_1', 'review_2', 'review_3']]
        # Rename the columns of the current evaluation
        current_evals.columns = [f'review_{review_column_count + 1}', f'review_{review_column_count + 2}', f'review_{review_column_count + 3}']
        # Merge the current evaluation with the merged_negated_evaluation
        merged_evaluation_df = pd.merge(merged_evaluation_df, current_evals, left_index=True, right_index=True)
    return merged_evaluation_df

def calculate_krippendorf_alpha(merged_evaluation_df, merged_evaluations_folder_path, dropped_column=None, was_dissimilar=False):
    three_category_ratings = np.zeros((len(merged_evaluation_df), 3))
    review_columns = [col for col in merged_evaluation_df.columns if 'review_' in col]
    for i, row in merged_evaluation_df.iterrows():
        for j in range(len(review_columns)):
            column_name = review_columns[j]
            review = int(float(row[column_name]))
            if review == 1 or review == 2:
                three_category_ratings[i, 0] += 1
            elif review == 3 or review == 4:
                three_category_ratings[i, 1] += 1
            else:
                three_category_ratings[i, 2] += 1
    three_category_alpha = krippendorff.alpha(three_category_ratings)
    # Save the results
    dropped_column_text = f'_dropped_{dropped_column}' if dropped_column != None else ''
    dissimilar_text = '_dissimilar' if was_dissimilar else ''
    save_alpha_filename = f'krippendorff_alpha_negated_{len(review_columns)}_reviews{dropped_column_text}{dissimilar_text}.txt'
    save_file_path = f'{merged_evaluations_folder_path}/{save_alpha_filename}'
    with open(save_file_path, 'w') as f:
        f.write(f'Alpha:\t{three_category_alpha}\n')
    print(f'Krippendorff alpha saved at: {save_file_path}')

def drop_the_most_dissimilar_review(merged_evaluation_df):
    # get review columns
    review_columns = [col for col in merged_evaluation_df.columns if 'review_' in col]
    if len(review_columns) > 0: 
        # extract the columns
        review_columns_df = merged_evaluation_df[review_columns]
        # calculate the similarity
        similarity = review_columns_df.corr()
        # Add all the values across columns
        similarity = similarity.sum(axis=1)
        # find least value
        least_similar_column = similarity.idxmin()
        # Drop the least similar column
        merged_evaluation_df = merged_evaluation_df.drop(columns=[least_similar_column])
    return merged_evaluation_df, least_similar_column

if __name__ == "__main__":

    """Usage example:
    python scripts/compare_methods.py
    """

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for experiment, folder_paths in tqdm(EXPERIMENTS.items(), desc='Experiments'):
        # Sanity check for the experiments existence
        if '' in folder_paths.values():
            print(f'*** Skipping {experiment} ***')
            print(f'*** Not all experiments are finished! ***')
            continue
        # Create folder for the experiment if not exists
        experiment_path = f'experiments/{experiment}'
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        # Constructing out_tsv file name and path
        # SageMaker review
        out_tsv_filename = f'compared_results.tsv'
        out_tsv_path = f'{experiment_path}/{out_tsv_filename}'
        # Majority vote
        out_tsv_filename_majority = f'compared_results_majority.tsv'
        out_tsv_path_majority = f'{experiment_path}/{out_tsv_filename_majority}'
        # Move out_tsv to a _bak file if exists
        if os.path.exists(out_tsv_path):
            os.system(f'mv {out_tsv_path} {experiment_path}/compared_results_bak.tsv')
        if os.path.exists(out_tsv_path_majority):
            os.system(f'mv {out_tsv_path_majority} {experiment_path}/compared_results_majority_bak.tsv')
        # Create the out_tsv file if not exists
        with open(out_tsv_path, 'w') as f:
            f.write('method\tsub_folder\tnormal_accuracy\tnegated_accuracy\n')
        with open(out_tsv_path_majority, 'w') as f:
            f.write('method\tsubfolder\tnormal_accuracy\tnegated_accuracy\n')
        
        # Post-process each method
        for method, folder_path in folder_paths.items():
            # Evaluation filenames
            negated_evaluation_filename = f'sampled_negated_preds_generated_{method}_evaluated.tsv'
            normal_evaluation_filename = f'sampled_normal_preds_generated_{method}_evaluated.tsv'

            # Empty daframes for the merged evaluations
            merged_negated_evaluation = pd.DataFrame()
            merged_normal_evaluation = pd.DataFrame()

            # Merged evaluation folder paths
            merged_evaluations_folder_path = f'{experiment_path}/{method}'
            if not os.path.exists(merged_evaluations_folder_path):
                os.makedirs(merged_evaluations_folder_path)

            # Merged evaluation file paths
            merged_negated_evaluation_path = f'{merged_evaluations_folder_path}/{negated_evaluation_filename}'
            merged_normal_evaluation_path = f'{merged_evaluations_folder_path}/{normal_evaluation_filename}'

            # Each method result has multiple subfolders resulting from different human evaluation rounds
            # We need to post-process each subfolder
            subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir() and '/mturk' not in f.path]
            neg_review_column_count, normal_review_column_count = 0, 0
            for subfolder in subfolders:
                eval_file_path = ''
                subfolder_name = subfolder.split('/')[-1]
                #region Loading the evaluations
                print(f'*** Evaluating {method} in {experiment} ***')
                # Negated
                # Check if subfolder is related to negated questions
                if '-negated-' in subfolder_name:
                    eval_file_path = f'{subfolder}/{negated_evaluation_filename}'
                    os.system(f'python scripts/simple_post_process.py --in_tsv {eval_file_path}')    
                # Normal
                # Check if subfolder is related to normal questions
                if '-normal-' in subfolder_name:
                    eval_file_path = f'{subfolder}/{normal_evaluation_filename}'
                    os.system(f'python scripts/simple_post_process.py --in_tsv {eval_file_path}')
                # Loading the evaluations as pandas dataframes
                eval_df = pd.read_csv(eval_file_path, sep='\t')
                #endregion

                #region Calculating the accuracy based on SageMaker review
                # Negated
                negated_json_evaluation_filename = f'sampled_negated_preds_generated_{method}_evaluated_results.json'
                negated_filepath = f'{subfolder}/{negated_json_evaluation_filename}'
                negated_accuracy = extract_accurcy_from_json(negated_filepath)
                # Normal
                normal_json_evaluation_filename = f'sampled_normal_preds_generated_{method}_evaluated_results.json'
                normal_filepath = f'{subfolder}/{normal_json_evaluation_filename}'
                normal_accuracy = extract_accurcy_from_json(normal_filepath)
                # Append the results to the out_tsv file
                with open(out_tsv_path, 'a') as f:
                    f.write(f'{method}\t{subfolder_name}\t{normal_accuracy}\t{negated_accuracy}\n')
                #endregion
            
                #region Calculating the accuracy based on majority vote
                # Negated
                negated_json_evaluation_filename_majority = f'sampled_negated_preds_generated_{method}_evaluated_results_majority.json'
                negated_majority_filepath = f'{subfolder}/{negated_json_evaluation_filename_majority}'
                negated_majority_accuracy = extract_accurcy_from_json(negated_majority_filepath)
                # Normal
                normal_json_evaluation_filename_majority = f'sampled_normal_preds_generated_{method}_evaluated_results_majority.json'
                normal_majority_filepath = f'{subfolder}/{normal_json_evaluation_filename_majority}'
                normal_majority_accuracy = extract_accurcy_from_json(normal_majority_filepath)
                # Append the results to the out_tsv file
                with open(out_tsv_path_majority, 'a') as f:
                    f.write(f'{method}\t{subfolder_name}\t{normal_majority_accuracy}\t{negated_majority_accuracy}\n')
                #endregion

                #region Merging the evaluations
                # Negated
                if '-negated-' in subfolder_name:
                    # Merge new evaluations with the previous ones
                    merged_negated_evaluation = add_new_vals_to_merged_evals(merged_negated_evaluation, eval_df, neg_review_column_count)
                    # Count columns with the name 'review_*'
                    neg_review_column_count += len([col for col in eval_df.columns if 'review_' in col])

                        
                # Normal
                if '-normal-' in subfolder_name:
                    # Merge new evaluations with the previous ones
                    merged_normal_evaluation = add_new_vals_to_merged_evals(merged_normal_evaluation, eval_df, normal_review_column_count)
                    # Count columns with the name 'review_*'
                    normal_review_column_count += len([col for col in eval_df.columns if 'review_' in col])
                #endregion
            
            #region Saving merged evaluations
            # Save the merged evaluations
            if len(merged_negated_evaluation) > 0:
                merged_negated_evaluation.to_csv(merged_negated_evaluation_path, sep='\t', index=False)
            if len(merged_normal_evaluation) > 0:
                merged_normal_evaluation.to_csv(merged_normal_evaluation_path, sep='\t', index=False)
            #endregion

            #region Calculating Krippendorff's alpha 
            # Negated
            # neg_review_column_count
            if len(merged_negated_evaluation) > 0:
                # Calculate Krippendorff's alpha
                calculate_krippendorf_alpha(merged_negated_evaluation, merged_evaluations_folder_path)
                # Drop the most dissimilar review
                updated_merged_evaluation, dropped_column_name = drop_the_most_dissimilar_review(merged_negated_evaluation)
                # Re-calculate Krippendorff's alpha
                calculate_krippendorf_alpha(updated_merged_evaluation, merged_evaluations_folder_path, dropped_column_name, was_dissimilar=True)
                # Dropping columns one by one
                review_columns = [col for col in merged_negated_evaluation.columns if 'review_' in col]
                for review_column in review_columns:
                    # Drop review_column
                    updated_merged_evaluation = merged_negated_evaluation.drop(columns=[review_column])
                    # Re-calculate Krippendorff's alpha
                    calculate_krippendorf_alpha(updated_merged_evaluation, merged_evaluations_folder_path, review_column)
            # Normal
            # TODO: normal
            # endregion