import argparse, os, json
from tqdm import tqdm

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
        # Create the out_tsv file if not exists
        if not os.path.exists(out_tsv_path):
            with open(out_tsv_path, 'w') as f:
                f.write('method\tsub_folder\tnormal_accuracy\tnegated_accuracy\n')
        if not os.path.exists(out_tsv_path_majority):
            with open(out_tsv_path_majority, 'w') as f:
                f.write('method\tsubfolder\tnormal_accuracy\tnegated_accuracy\n')
        
        # Post-process each method
        for method, folder_path in folder_paths.items():

            # Each method result has multiple subfolders resulting from different human evaluation rounds
            # We need to post-process each subfolder
            subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir() and '/mturk' not in f.path]
            for subfolder in subfolders:
                subfolder_name = subfolder.split('/')[-1]
                #region Loading the evaluations
                print(f'*** Evaluating {method} in {experiment} ***')
                # Negated
                # Check if subfolder is related to negated questions
                if '-negated-' in subfolder_name:
                    negated_evaluation_filename = f'sampled_negated_preds_generated_{method}_evaluated.tsv'
                    os.system(f'python scripts/simple_post_process.py --in_tsv {subfolder}/{negated_evaluation_filename}')    
                # Normal
                # Check if subfolder is related to normal questions
                if '-normal-' in subfolder_name:
                    normal_evaluation_filename = f'sampled_normal_preds_generated_{method}_evaluated.tsv'
                    os.system(f'python scripts/simple_post_process.py --in_tsv {subfolder}/{normal_evaluation_filename}')
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