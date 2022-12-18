import argparse, os

if __name__ == "__main__":
    """ Sample AWS CLI command to download the results. We need to go to mturk results folder and run this command:
    aws s3 cp --recursive s3://negated-predicates/self_samples/ ./
    aws s3 cp --recursive s3://[folder-path-to-results] ./
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--mturk_path", type=str, default=f'./experiments/test_experiment/mturk/samples-negated')
    parser.add_argument("--out_tsv", type=str, default=f'./experiments/test_experiment/sample.tsv')
    args = parser.parse_args()

    # Post-process the results
    print('*** Post-processing the results ***')
    os.system(f'python scripts/post_process_mturk_evaluations.py --mturk_path {args.mturk_path} --out_tsv {args.out_tsv}')
    # Plot the results
    print('*** Plotting the results ***')
    os.system(f'python scripts/plot_evaluated_results.py --in_tsv {args.out_tsv}')