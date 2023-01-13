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

if __name__ == "__main__":

    """Usage example:
    python scripts/compare_methods.py
    """

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    for experiment, folder_paths in tqdm(EXPERIMENTS.items(), desc='Experiments'):
        # Sanity check
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
                f.write('method\tnormal_accuracy\tnegated_accuracy\n')
        if not os.path.exists(out_tsv_path_majority):
            with open(out_tsv_path_majority, 'w') as f:
                f.write('method\tnormal_accuracy\tnegated_accuracy\n')
        
        # Post-process each method
        for method, folder_path in folder_paths.items():
            print(f'*** Evaluating {method} in {experiment} ***')
            # Negated
            negated_evaluation_filename = f'sampled_negated_preds_generated_{method}_evaluated.tsv'
            os.system(f'python scripts/simple_post_process.py --in_tsv {folder_path}/{negated_evaluation_filename}')    
            # Normal
            normal_evaluation_filename = f'sampled_normal_preds_generated_{method}_evaluated.tsv'
            os.system(f'python scripts/simple_post_process.py --in_tsv {folder_path}/{normal_evaluation_filename}')
            # Get accuracy of the method from the evaluation JSON file sampled_negated_preds_generated_cot_qa_evaluated_results.json
            
            # Open the evaluation JSON file
            # Negated
            negated_json_evaluation_filename = f'sampled_negated_preds_generated_{method}_evaluated_results.json'
            negated_accuracy, normal_accuracy = 0, 0
            with open(f'{folder_path}/{negated_json_evaluation_filename}', 'r') as f:
                negated_results = json.load(f)
                negated_accuracy = int(negated_results['total']['correct'])*100.0/int(negated_results['total']['count'])
            # Normal
            normal_json_evaluation_filename = f'sampled_normal_preds_generated_{method}_evaluated_results.json'
            with open(f'{folder_path}/{normal_json_evaluation_filename}', 'r') as f:
                normal_results = json.load(f)
                normal_accuracy = int(normal_results['total']['correct'])*100.0/int(normal_results['total']['count'])
            # Append the results to the out_tsv file
            with open(out_tsv_path, 'a') as f:
                f.write(f'{method}\t{normal_accuracy}\t{negated_accuracy}\n')
            
            # Majority vote
            # Negated
            negated_json_evaluation_filename_majority = f'sampled_negated_preds_generated_{method}_evaluated_results_majority.json'
            majority_negated_accuracy, majority_normal_accuracy = 0, 0
            with open(f'{folder_path}/{negated_json_evaluation_filename_majority}', 'r') as f:
                negated_results = json.load(f)
                majority_negated_accuracy = int(negated_results['total']['correct'])*100.0/int(negated_results['total']['count'])
            # Normal
            normal_json_evaluation_filename_majority = f'sampled_normal_preds_generated_{method}_evaluated_results_majority.json'
            with open(f'{folder_path}/{normal_json_evaluation_filename_majority}', 'r') as f:
                normal_results = json.load(f)
                majority_normal_accuracy = int(normal_results['total']['correct'])*100.0/int(normal_results['total']['count'])
            # Append the results to the out_tsv file
            with open(out_tsv_path_majority, 'a') as f:
                f.write(f'{method}\t{majority_normal_accuracy}\t{majority_negated_accuracy}\n')