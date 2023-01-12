import argparse, sys, traceback
import pandas as pd
from tqdm import tqdm
sys.path.append('./')
from utils.atomic_utils import verbalize_subject_predicate, atomic_preds, limited_atomic_preds

if __name__ == "__main__":
    print('Loading spacy ...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str, default="atomic2020", choices=["conceptnet", "transomcs", "atomic", "atomic2020", "wpkg", "wpkg_expanded", "negated_cs"])
    # If all, use -1 for size_per_predicate
    parser.add_argument("--size_per_predicate", type=int, default=10)
    parser.add_argument("--input", type=str, default="data/atomic2020/test.tsv")
    parser.add_argument("--experiment_path", type=str, default="experiments/atomic_2020_eval")
    parser.add_argument("--limited_preds", action="store_true")
    # Preds variable name. It can be found under atomic_utils.py
    parser.add_argument("--preds_var_name", type=str, default="limited_atomic_preds")
    args = parser.parse_args()
    seed = 66

    if args.kg not in ["atomic2020", "negated_cs"]:
        raise NotImplementedError("Only atomic2020 is supported for now")

    # Out file paths
    negated_samples_file_path = f'{args.experiment_path}/sampled_negated_preds.tsv'
    normal_samples_file_path = f'{args.experiment_path}/sampled_normal_preds.tsv'
    # Load args.input as a pandas dataframe
    df = pd.read_csv(args.input, sep="\t", header=None, names=["head", "relation", "tail"])
    # Remove dupicate rows based on head and relation columns
    df = df.drop_duplicates(subset=['head', 'relation'])
    # Remove all rows that have string "___" in the head column
    df = df[~df['head'].str.contains("___")]
    # Empty dataframe to store sampled rows
    all_normal_sampled_df = pd.DataFrame(columns=['head', 'relation', 'tail', 'prompt'])
    all_negated_sampled_df = pd.DataFrame(columns=['head', 'relation', 'tail', 'prompt'])
    # Choose the preds to use
    if args.kg == "atomic2020":
        try:
            preds = atomic_preds if not args.limited_preds else eval(args.preds_var_name)
        except:
            # Print the line that caused the exception
            print('Variable with this name not found {}. Maybe you need to import or define it!'.format(args.preds_var_name))
            traceback.print_exc()
    else:
        # Get list of all unique relations in df
        preds = df['relation'].unique().tolist()
        if 'relation' in preds:
            preds.remove('relation')
    # Remove isFilledBy from atomic_preds as it is not a valid relation (all include "___")
    if 'isFilledBy' in preds:
        preds.remove('isFilledBy')
    # Also removing NotDesires as we result in that in negation
    if 'NotDesires' in preds:
        preds.remove('NotDesires')
    # Iterate over all atomic predicates
    progress_bar = tqdm(preds)
    for pred in progress_bar:
        negated_pred = f'Not{pred}'
        progress_bar.set_description("Processing {}".format(pred))
        # Exception handling if there is an error in the following lines
        try:
            # Sample args.size_per_predicate rows with predicate pred
            if args.size_per_predicate == -1:
                sampled_df = df[df['relation'] == pred]
            else:
                sampled_df = df[df['relation'] == pred].sample(n=args.size_per_predicate, random_state=seed)
            sampled_df_negated = sampled_df.copy()
            # Change all relation values to negated_pred in sampled_df_negated
            sampled_df_negated['relation'] = negated_pred
            # Add a new column "prompt" to sampled_df and sampled_df_negated
            sampled_df['prompt'] = sampled_df.apply(lambda row: verbalize_subject_predicate(args.kg, row.to_dict()), axis=1)
            sampled_df_negated['prompt'] = sampled_df_negated.apply(lambda row: verbalize_subject_predicate(args.kg, row.to_dict()), axis=1)
            # Append sampled_df to all_sampled_df
            all_normal_sampled_df = all_normal_sampled_df.append(sampled_df)
            all_negated_sampled_df = all_negated_sampled_df.append(sampled_df_negated)
        except:
            print('Error in {}'.format(pred))
            import IPython; IPython. embed(); exit(1)

    # Save all_sampled_df to args.output
    all_normal_sampled_df.to_csv(normal_samples_file_path, sep="\t", index=False, header=True)
    all_negated_sampled_df.to_csv(negated_samples_file_path, sep="\t", index=False, header=True)