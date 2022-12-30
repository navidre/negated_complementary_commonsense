import argparse, os
from pathlib import Path

if __name__ == "__main__":

    """Usage example:
    For evaluations done using Amazon SageMaker Ground Truth:
    python scripts/simple_post_process.py --s3_path s3://negated-predicates/sampled_10_atomic2020_few_shot_limited_preds/negated-few-shot-limited/ --evaluated_jsonl_filename sampled_negated_preds_generated_few_shot_mturk.jsonl

    For local evaluation files:
    python scripts/simple_post_process.py --in_tsv experiments/sampled_10_atomic2020_few_shot_limited_preds/sampled_negated_preds_generated_few_shot_qa_evaluated.tsv
    """

    parser = argparse.ArgumentParser()
    # S3 path to the results. Can be copied from the Labeling job's console, under "Output dataset location".
    parser.add_argument("--s3_path", type=str, default=f's3://negated-predicates/sampled_10_atomic2020_few_shot_limited_preds/negated-few-shot-limited/')
    # The TSV generation file, which was converted to JSONL and uploaded to S3 for mTurk evaluation.
    # The name can be see under Labeling job's console, under "Input dataset location".
    # Use the relative local path.
    parser.add_argument("--evaluated_jsonl_filename", type=str, default=f'sampled_negated_preds_generated_few_shot_mturk.jsonl')
    # The evaluated TSV file name. Use the relative local path.
    parser.add_argument("--in_tsv", type=str, default=f'')
    args = parser.parse_args()

    if args.in_tsv == '':
        # This is the case to download the results from S3 and post-process them
        # Extracting experiment name from s3_path and the constructing experiment folder
        experiment_name = str(Path(args.s3_path).parent).split('/')[-1]
        experiment_folder = f'experiments/{experiment_name}'

        # Constructing out_tsv file name and path
        out_tsv_filename = os.path.basename(args.evaluated_jsonl_filename).replace('_mturk.jsonl', '_evaluated.tsv')
        out_tsv_path = f'{experiment_folder}/{out_tsv_filename}'

        # Download the results
        print('*** Downloading the results ***')
        experiment_name = args.s3_path.split('/')[-2]
        # Create mturk folder if not exists
        mturk_parent_path = f'{experiment_folder}/mturk'
        if not os.path.exists(mturk_parent_path):
            os.makedirs(mturk_parent_path)
        mturk_path = f'{mturk_parent_path}/{experiment_name}'
        if not os.path.exists(mturk_path):
            os.makedirs(mturk_path)
        os.system(f'aws s3 cp --recursive {args.s3_path} {mturk_path}')

        # Post-process the results
        print('*** Post-processing the results ***')
        os.system(f'python scripts/post_process_mturk_evaluations.py --mturk_path {mturk_path} --out_tsv {out_tsv_path}')
    else:
        out_tsv_path = args.in_tsv
    
    # Plot the results
    print('*** Plotting the results ***')
    os.system(f'python scripts/plot_evaluated_results.py --in_tsv {out_tsv_path}')