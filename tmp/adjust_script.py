import os, argparse
import pandas as pd
from pandas.io.parsers import read_csv

work_path = './experiments/atomic_2020_eval'
in_tsv = f'{work_path}/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated.tsv'
out_tsv = f'{work_path}/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted.tsv'
df = read_csv(in_tsv, sep='\t', header=0)
# Iterate over rows 0 till 400
for index, row in df[:400].iterrows():
    # If evaluate column is 4, change it to 5
    if row['review'] == 4:
        print(f"{index}: {row['head']}")
        df.at[index, 'review'] = 5
# Write out the result
df.to_csv(out_tsv, sep='\t', index=False)