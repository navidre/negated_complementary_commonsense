import argparse, sys, os

# Some constants
S3_BUCKET = 'negated-predicates'
DATASET_FILE = {'atomic2020': 'data/atomic2020/test.tsv'}

if __name__ == "__main__":
    """Sample command runs
    - Limited predicates:
    python scripts/simple_pre_process_and_generate.py --method few_shot --kg atomic2020 --size_per_predicate 10 --limited_preds
    - All predicates:
    python scripts/simple_pre_process_and_generate.py --method few_shot --kg atomic2020 --size_per_predicate 10
    """
    parser = argparse.ArgumentParser()
    # Generation method
    parser.add_argument("--method", type=str, default="few_shot", choices=["few_shot", "cot"])
    # Knowledge graph to use
    parser.add_argument("--kg", type=str, default="atomic2020", choices=["conceptnet", "transomcs", "atomic", "atomic2020", "wpkg", "wpkg_expanded"])
    # Number of subject-object pairs to sample per predicate
    parser.add_argument("--size_per_predicate", type=int, default=10)
    # Selectively choosing limited predicates, especially the worse ones to test
    parser.add_argument("--limited_preds", action="store_true")
    parser.add_argument("--num_generations", type=int, default=3)
    # parser.add_argument("--output", type=str, default="experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv")
    # parser.add_argument("--negated", action="store_true", default=True)
    args = parser.parse_args()

    if args.kg != "atomic2020":
        raise NotImplementedError("Only atomic2020 is supported for now")

    # Create experiments folder
    print('*** Creating experiments folder ***')
    limited_preds_str = '_limited_preds' if args.limited_preds else ''
    experiment_name = f'sampled_{args.size_per_predicate}_{args.kg}_{args.method}{limited_preds_str}'
    experiment_path = f'experiments/{experiment_name}'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    
    # Sample normal and negated predicates
    print('*** Sampling normal and negated predicates ***')
    if not os.path.exists(f'{experiment_path}/sampled_normal_preds.tsv'):
        limited_preds_arg_str = '--limited_preds' if args.limited_preds else ''
        input_file_path = DATASET_FILE[args.kg]
        os.system(f'python scripts/prepare_subjects_preds_for_generation.py --kg {args.kg} --size_per_predicate {args.size_per_predicate} --input {input_file_path} --experiment_path {experiment_path} {limited_preds_arg_str}')
    
    assert os.path.exists(f'{experiment_path}/sampled_normal_preds.tsv'), f'File {experiment_path}/sampled_normal_preds.tsv does not exist. Sampled predicates are not generated.'
    assert os.path.exists(f'{experiment_path}/sampled_negated_preds.tsv'), f'File {experiment_path}/sampled_negated_preds.tsv does not exist. Sampled predicates are not generated.'

    # Generate response using the method args.method
    print(f'*** Generating responses using the method {args.method} ***')
    if not os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}.tsv'):
        os.system(f'python scripts/generate_objects_using_gpt_3.py --input {experiment_path}/sampled_normal_preds.tsv --style {args.method} --num_generations {args.num_generations}')
    if not os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}.tsv'):
        os.system(f'python scripts/generate_objects_using_gpt_3.py --input {experiment_path}/sampled_negated_preds.tsv --style {args.method} --negated --num_generations {args.num_generations}')
    
    assert os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}.tsv'), f'File {experiment_path}/sampled_normal_preds_generated_{args.method}.tsv does not exist. Generated responses are not generated.'
    assert os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}.tsv'), f'File {experiment_path}/sampled_negated_preds_generated_{args.method}.tsv does not exist. Generated responses are not generated.'

    # Make JSONL files ready
    print('*** Making JSONL files ready ***')
    if not os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}_evaluated.tsv'):
        os.system(f'python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv {experiment_path}/sampled_normal_preds_generated_{args.method}.tsv')
    if not os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}_evaluated.tsv'):
        os.system(f'python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv {experiment_path}/sampled_negated_preds_generated_{args.method}.tsv')

    assert os.path.exists(f'{experiment_path}/sampled_normal_preds_generated_{args.method}_evaluated.tsv'), f'File {experiment_path}/mturk/sampled_normal_preds_generated_{args.method}_evaluated.tsv does not exist. File template for evaluations is not generated.'
    assert os.path.exists(f'{experiment_path}/sampled_negated_preds_generated_{args.method}_evaluated.tsv'), f'File {experiment_path}/mturk/sampled_negated_preds_generated_{args.method}_evaluated.tsv does not exist. File template for evaluations is not generated.'

    # Upload the JSONL files to the S3 bucket in an appropriate folder
    os.system(f'aws s3 cp {experiment_path}/mturk/sampled_normal_preds_generated_few_shot_mturk.jsonl s3://{S3_BUCKET}/{experiment_name}/')
    os.system(f'aws s3 cp {experiment_path}/mturk/sampled_negated_preds_generated_few_shot_mturk.jsonl s3://{S3_BUCKET}/{experiment_name}/')