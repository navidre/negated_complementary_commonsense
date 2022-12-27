import argparse, os

if __name__ == "__main__":

    """Usage example:
    python scripts/simple_post_process.py --s3_path s3://negated-predicates/sampled_10_atomic2020_cot_qa_limited_preds/negated/ --experiment_folder ./experiments/sampled_10_atomic2020_cot_qa_limited_preds --out_tsv experiments/sampled_10_atomic2020_cot_qa_limited_preds/sampled_negated_preds_generated_cot_qa_evaluated.tsv
    """

    parser = argparse.ArgumentParser()
    # S3 path to the results. Can be copied from the Labeling job's console, under "Output dataset location".
    parser.add_argument("--s3_path", type=str, default=f's3://negated-predicates/self_samples/sample-negated/results/samples-negated/')
    # experiment_folder is the experiment folder, including the TSV generation file, which was converted to JSONL and uploaded to S3 for mTurk evaluation.
    parser.add_argument("--experiment_folder", type=str, default=f'./experiments/test_experiment')
    # The TSV generation file, which was converted to JSONL and uploaded to S3 for mTurk evaluation.
    # The name can be see under Labeling job's console, under "Input dataset location".
    # Use the relative local path.
    parser.add_argument("--out_tsv", type=str, default=f'./experiments/test_experiment/sample.tsv')
    args = parser.parse_args()


    # Download the results
    print('*** Downloading the results ***')
    experiment_name = args.s3_path.split('/')[-2]
    # Create mturk folder if not exists
    mturk_parent_path = f'{args.experiment_folder}/mturk'
    if not os.path.exists(mturk_parent_path):
        os.makedirs(mturk_parent_path)
    mturk_path = f'{mturk_parent_path}/{experiment_name}'
    if not os.path.exists(mturk_path):
        os.makedirs(mturk_path)
    os.system(f'aws s3 cp --recursive {args.s3_path} {mturk_path}')

    # Post-process the results
    print('*** Post-processing the results ***')
    os.system(f'python scripts/post_process_mturk_evaluations.py --mturk_path {mturk_path} --out_tsv {args.out_tsv}')
    # Plot the results
    print('*** Plotting the results ***')
    os.system(f'python scripts/plot_evaluated_results.py --in_tsv {args.out_tsv}')