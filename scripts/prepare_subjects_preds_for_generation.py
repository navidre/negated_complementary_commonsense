import argparse, sys
import pandas as pd
from tqdm import tqdm
sys.path.append('./')
from utils.atomic_utils import verbalize_subject_predicate, negated_atomic_preds, atomic_preds

if __name__ == "__main__":
    print('Loading spacy ...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=str, default="atomic2020", choices=["conceptnet", "transomcs", "atomic", "atomic2020", "wpkg", "wpkg_expanded"])
    parser.add_argument("--input", type=str, default="data/atomic2020/test.tsv")
    parser.add_argument("--output", type=str, default="experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv")
    parser.add_argument("--negated", action="store_true", default=True)
    args = parser.parse_args()
    seed = 66

    # Load args.input as a pandas dataframe
    df = pd.read_csv(args.input, sep="\t", header=None, names=["head", "relation", "tail"])
    # Remove dupicate rows based on head and relation columns
    df = df.drop_duplicates(subset=['head', 'relation'])
    # Remove all rows that have string "___" in the head column
    df = df[~df['head'].str.contains("___")]
    # Empty dataframe to store sampled rows
    all_sampled_df = pd.DataFrame(columns=['head', 'relation', 'tail', 'prompt'])
    # Remove isFilledBy from atomic_preds as it is not a valid relation (all include "___")
    atomic_preds.remove('isFilledBy')
    # Also removing NotDesires as we result in that in negation
    atomic_preds.remove('NotDesires')
    # Iterate over all atomic predicates
    progress_bar = tqdm(atomic_preds)
    for pred in progress_bar:
        negated_pred = f'Not{pred}'
        progress_bar.set_description("Processing {}".format(pred))
        # Exception handling if there is an error in the following lines
        try:
            # Sample 10 rows with predcate "NotDesires"
            sampled_df = df[df['relation'] == pred].sample(n=10, random_state=seed)
            # Change all relation values to negated_pred
            sampled_df['relation'] = negated_pred
            # Add a new column "prompt" to sampled_df
            sampled_df['prompt'] = sampled_df.apply(lambda row: verbalize_subject_predicate(args.kg, row.to_dict()), axis=1)
            # Append sampled_df to all_sampled_df
            all_sampled_df = all_sampled_df.append(sampled_df)
        except:
            print('Error in {}'.format(pred))
            import IPython; IPython. embed(); exit(1)

    # Save all_sampled_df to args.output
    all_sampled_df.to_csv(args.output, sep="\t", index=False, header=True)