import argparse, sys, os

# Some constants
S3_BUCKET = 'negated-predicates'
DATASET_FILE = {'atomic2020': 'data/atomic2020/test.tsv',
                'visualcomet': 'data/visualcomet/test_annots.json',
                'negated_cs': 'data/negated_cs/data.tsv'
}

if __name__ == "__main__":
    """Sample command runs
    - Limited predicates:
    python scripts/simple_pre_process_and_generate.py --method few_shot --kg atomic2020 --size_per_predicate 10 --limited_preds --preds_var_name limited_atomic_preds
    - No size limit per predicate:
    python scripts/simple_pre_process_and_generate.py --method few_shot --kg negated_cs --size_per_predicate -1
    - All predicates:
    python scripts/simple_pre_process_and_generate.py --method few_shot --kg atomic2020 --size_per_predicate 10
    - Ten limited atomic preds with 10 samples each:
    python scripts/simple_pre_process_and_generate.py --method cot_qa_updated_neg_teach_var_temp --kg atomic2020 --size_per_predicate 10 --limited_preds --preds_var_name ten_atomic_preds
    """

    #region Argument parsing
    parser = argparse.ArgumentParser()
    # Generation method
    parser.add_argument("--method", type=str, default="few_shot", choices=["few_shot", "cot_qa", "few_shot_qa", "updated_cot_qa", "cot_qa_neg_teach", "cot_qa_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp"])
    # Knowledge graph to use
    parser.add_argument("--kg", type=str, default="atomic2020", choices=["conceptnet", "transomcs", "atomic", "atomic2020", "wpkg", "wpkg_expanded", "visualcomet", "negated_cs"])
    # Number of subject-object pairs to sample per predicate
    # If all, use -1
    parser.add_argument("--size_per_predicate", type=int, default=10)
    # Selectively choosing limited predicates, especially the worse ones to test
    # Limited preds is only supported for atomic2020 at the moment
    parser.add_argument("--limited_preds", action="store_true")
    # Preds variable name. It can be found under atomic_utils.py
    parser.add_argument("--preds_var_name", type=str, default="limited_atomic_preds")
    # Number of generations to make for each subject-predicate pair
    parser.add_argument("--num_generations", type=int, default=3)
    args = parser.parse_args()
    #endregion

    #region Checking if the methods or args are supported
    if args.kg not in ["atomic2020", "visualcomet", "negated_cs"]:
        raise NotImplementedError("The KG is not supported.")
    if args.kg != "atomic2020" and args.limited_preds:
        raise NotImplementedError("Limited predicates are only supported for atomic2020.")
    #endregion

    #region Creating experiments folder
    print('*** Creating experiments folder ***')
    limited_preds_str = '_limited_preds' if args.limited_preds else ''
    preds_name_str = f'_{args.preds_var_name}' if args.limited_preds else ''
    experiment_name = f'sampled_{args.size_per_predicate}_{args.kg}_{args.method}{limited_preds_str}{preds_name_str}'
    experiment_path = f'experiments/{experiment_name}'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    #endregion
    
    #region Sampling normal and negated predicates
    print('*** Sampling normal and negated predicates ***')
    if not os.path.exists(f'{experiment_path}/sampled_normal_preds.tsv') or not os.path.exists(f'{experiment_path}/sampled_negated_preds.tsv'):
        limited_preds_arg_str = '--limited_preds' if args.limited_preds else ''
        preds_name_var_str = f'--preds_var_name {args.preds_var_name}' if args.limited_preds else ''
        input_file_path = DATASET_FILE[args.kg]
        os.system(f'python scripts/prepare_subjects_preds_for_generation.py --kg {args.kg} --size_per_predicate {args.size_per_predicate} --input {input_file_path} --experiment_path {experiment_path} {limited_preds_arg_str} {preds_name_var_str}')
    
    assert os.path.exists(f'{experiment_path}/sampled_normal_preds.tsv'), f'*** File {experiment_path}/sampled_normal_preds.tsv does not exist. Sampled predicates are not generated.'
    assert os.path.exists(f'{experiment_path}/sampled_negated_preds.tsv'), f'*** File {experiment_path}/sampled_negated_preds.tsv does not exist. Sampled predicates are not generated.'
    #endregion

    #region Generating response using the method args.method
    print(f'*** Generating responses using the method {args.method} ***')
    if not os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}.tsv'):
        os.system(f'python scripts/generate_objects_using_gpt_3.py --input {experiment_path}/sampled_normal_preds.tsv --style {args.method} --num_generations {args.num_generations}')
    if not os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}.tsv'):
        os.system(f'python scripts/generate_objects_using_gpt_3.py --input {experiment_path}/sampled_negated_preds.tsv --style {args.method} --negated --num_generations {args.num_generations}')
    
    assert os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}.tsv'), f'*** File {experiment_path}/sampled_normal_preds_generated_{args.method}.tsv does not exist. Generated responses are not generated.'
    assert os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}.tsv'), f'*** File {experiment_path}/sampled_negated_preds_generated_{args.method}.tsv does not exist. Generated responses are not generated.'
    #endregion

    #region Automatically evaluating the generated responses and saving the results
    print('*** Automatically evaluating the generated responses and saving the results ***')
    # Normal
    if not os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}_evaluated.tsv'):
        os.system(f'python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv {experiment_path}/sampled_normal_preds_generated_{args.method}.tsv --action auto_evaluate')
    assert os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}_evaluated.tsv'), f'*** File {experiment_path}/sampled_normal_preds_generated_{args.method}_evaluated.tsv does not exist. File template for evaluations is not generated.'

    # Negated
    if not os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}_evaluated.tsv'):
        os.system(f'python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv {experiment_path}/sampled_negated_preds_generated_{args.method}.tsv --action auto_evaluate')
    assert os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}_evaluated.tsv'), f'*** File {experiment_path}/sampled_negated_preds_generated_{args.method}_evaluated.tsv does not exist. File template for evaluations is not generated.'
    #endregion
    
    #region Making JSONL files ready
    print('*** Making JSONL files ready ***')
    # Normal
    if not os.path.exists(f'{experiment_path}/mturk/sampled_normal_preds_generated_{args.method}_mturk.jsonl'):
        os.system(f'python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv {experiment_path}/sampled_normal_preds_generated_{args.method}.tsv --action generate_jsonl')
    assert os.path.exists(f'{experiment_path}/mturk/sampled_normal_preds_generated_{args.method}_mturk.jsonl'), f'*** File {experiment_path}/mturk/sampled_normal_preds_generated_{args.method}_mturk.jsonl does not exist. JSONL file is not generated.'
    
    # Negated
    if not os.path.exists(f'{experiment_path}/mturk/sampled_negated_preds_generated_{args.method}_evaluated.tsv'):
        os.system(f'python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv {experiment_path}/sampled_negated_preds_generated_{args.method}.tsv --action generate_jsonl')
    assert os.path.exists(f'{experiment_path}/mturk/sampled_negated_preds_generated_{args.method}_mturk.jsonl'), f'*** File {experiment_path}/mturk/sampled_negated_preds_generated_{args.method}_mturk.jsonl does not exist. JSONL file is not generated.'
    #endregion

    #region Upload the JSONL files to the S3 bucket in an appropriate folder
    print(f'*** Uploading the JSONL files to the S3 bucket in a folder named {experiment_name} under {S3_BUCKET} bucket ***')
    
    if os.path.getsize(f'{experiment_path}/mturk/sampled_normal_preds_generated_{args.method}_mturk.jsonl') == 0:
        raise Exception(f'*** File {experiment_path}/mturk/sampled_normal_preds_generated_{args.method}_mturk.jsonl is empty. Skipping upload.')
    if os.path.getsize(f'{experiment_path}/mturk/sampled_negated_preds_generated_{args.method}_mturk.jsonl') == 0:
        raise Exception(f'*** File {experiment_path}/mturk/sampled_negated_preds_generated_{args.method}_mturk.jsonl is empty. Skipping upload.')
    
    os.system(f'aws s3 cp {experiment_path}/mturk/sampled_normal_preds_generated_{args.method}_mturk.jsonl s3://{S3_BUCKET}/{experiment_name}/')
    os.system(f'aws s3 cp {experiment_path}/mturk/sampled_negated_preds_generated_{args.method}_mturk.jsonl s3://{S3_BUCKET}/{experiment_name}/')
    #endregion