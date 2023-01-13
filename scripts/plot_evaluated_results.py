import os, argparse, json
import pandas as pd
from pandas.io.parsers import read_csv
# Plot results
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def count_annotations(in_tsv):
    # Extracting file name
    filename = os.path.basename(in_tsv).split('.')[0]
    out_filename = f'{filename}_results'
    out_filename_majority = f'{filename}_results_majority'
    work_path = os.path.dirname(args.in_tsv)
    out_json = os.path.join(work_path, f'{out_filename}.json')
    out_json_majority = os.path.join(work_path, f'{out_filename_majority}.json')
    # read tsv file with columns head, relation, prompt, generated_tail, full_text
    df = read_csv(in_tsv, sep='\t', header=0)
    # Calculate review accuracy by relation. Have count and unique values
    # df_review = df.groupby(['relation']).agg({'review': ['count', 'mean']})
    # Group df by relation
    df_review = df.groupby(['relation'])
    results = {}
    majority_results = {}
    total_count, total_correct, total_incorrect, total_unfamiliar = 0, 0, 0, 0
    total_count_majority, total_correct_majority, total_incorrect_majority, total_unfamiliar_majority = 0, 0, 0, 0
    for name, group in df_review:
        total, count_correct, count_incorrect, count_unfamiliar = group['review'].count(), 0, 0, 0
        if 'majority_vote' not in group.columns:
            group['majority_vote'] = 0
        total_majority, count_correct_majority, count_incorrect_majority, count_unfamiliar_majority = group['majority_vote'].count(), 0, 0, 0
        # Count unfamiliar as review 5
        for index, row in group.iterrows():
            # Count review
            if row['review'] == 1 or row['review'] == 2:
                count_correct += 1
            elif row['review'] == 3 or row['review'] == 4:
                count_incorrect += 1
            elif row['review'] == 5:
                count_unfamiliar += 1
            # Count majority vote
            if row['majority_vote'] == 1 or row['majority_vote'] == 2:
                count_correct_majority += 1
            elif row['majority_vote'] == 3 or row['majority_vote'] == 4:
                count_incorrect_majority += 1
            elif row['majority_vote'] == 5:
                count_unfamiliar_majority += 1
        # Count review totals
        total_count += total
        total_correct += count_correct
        total_incorrect += count_incorrect
        total_unfamiliar += count_unfamiliar
        # Count majority vote totals
        total_count_majority += total_majority
        total_correct_majority += count_correct_majority
        total_incorrect_majority += count_incorrect_majority
        total_unfamiliar_majority += count_unfamiliar_majority
        # Store results
        results[name] = {'count': total,
                         'correct': count_correct,
                         'incorrect': count_incorrect,
                         'unfamiliar': count_unfamiliar
                        }
        majority_results[name] = {'count': total_majority,
                                  'correct': count_correct_majority,
                                  'incorrect': count_incorrect_majority,
                                  'unfamiliar': count_unfamiliar_majority
                                 }
    # Add totals
    results['total'] = {'count': total_count,
                        'correct': total_correct,
                        'incorrect': total_incorrect,
                        'unfamiliar': total_unfamiliar
                        }
    majority_results['total'] = {'count': total_count_majority,
                                 'correct': total_correct_majority,
                                 'incorrect': total_incorrect_majority,
                                 'unfamiliar': total_unfamiliar_majority
                                 }
    # Save results as JSON file in work_path directory 
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    with open(out_json_majority, 'w') as f:
        json.dump(majority_results, f, indent=4, default=str)
    # Return results
    return results, majority_results

def plot_the_results(results,  majority_results, in_tsv):
    print(f"Plotting results from {in_tsv}")
    # Extracting file name
    filename = os.path.basename(in_tsv).split('.')[0]
    out_filename = f'{filename}_results'
    out_filename_majority = f'{filename}_results_majority'
    work_path = os.path.dirname(args.in_tsv)
    # DataFrame from results dictionary
    dfs = [pd.DataFrame.from_dict(results, orient='index'), pd.DataFrame.from_dict(majority_results, orient='index')]
    out_pdfs = [os.path.join(work_path, f'{out_filename}.pdf'), os.path.join(work_path, f'{out_filename_majority}.pdf')]
    index = 0
    for df in dfs:
        # Divide each row by count column to get percentages
        df_norm = df.div(df['count'], axis=0)
        # Set index as column 'relation'
        df_norm = df_norm.reset_index()
        # Sort by incorrect column
        df_norm = df_norm.sort_values(by=['incorrect'], ascending=False)
        # Plot
        df_norm.plot(x='index', y=['correct', 'incorrect', 'unfamiliar'], kind='bar')
        # Set title
        plt.title('Review accuracy by relation')
        # Set x label
        plt.xlabel('Relation')
        # Set y label
        plt.ylabel('Percentage')
        # Make plot wider
        plt.gcf().set_size_inches(20, 10)
        # Rotate x labels by 45 degrees
        plt.xticks(rotation=45, ha = 'right')
        # Increase font size
        plt.rcParams.update({'font.size': 22})
        # Save plot as pdf
        plt.savefig(out_pdfs[index], bbox_inches='tight')
        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tsv", type=str, default=f'./experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted.tsv')
    args = parser.parse_args()

    """
    Examples calls:
    - Normal predicates:
    python scripts/plot_evaluated_results.py --in_tsv experiments/atomic_2020_eval/few_shot_sampled_to_eval_with_gpt_3_self_evaluated.tsv
    """

    results, majority_results = count_annotations(args.in_tsv)
    plot_the_results(results, majority_results, args.in_tsv)