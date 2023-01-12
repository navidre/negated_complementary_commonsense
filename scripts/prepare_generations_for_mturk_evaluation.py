import argparse, os, sys, traceback
import pandas as pd
from tqdm import tqdm
import language_tool_python
sys.path.append('./')
from human_evaluate_generations import auto_evaluate_row

FIX_GRAMMER = True

INSTRUCTION_NOTES = "INSTRUCTION NOTES:\\n1. Instead of names, PersonX and PersonY are used to be gender neutral.\\n2. Please ignore grammatical errors and focus on commonsense.\\n3. If response is vague, such as 'not fireman', or a random word that does not fit the scenario, please choose 4 (not enough information).\\n\\nEvaluate this based on your commonsense:\\n"

def generate_jsonl_for_mturk(in_file_path, out_jsonl_path):
    # read tsv file with columns head, relation, prompt, generated_tail, full_text
    df = pd.read_csv(in_file_path, sep='\t', header=0)
    # Opening output file for writing
    jsonl_file = open(out_jsonl_path, 'w')
    # Interate over rows and generate jsonl file
    for index, row in tqdm(df.iterrows(), desc="Generating jsonl file", total=len(df)):
        try:
            # skip if already evaluated
            if row['review_1'] != 0:
                continue
            # Add to jsonl file
            full_text = row['full_text'] if not FIX_GRAMMER else easy_fix_grammar(row['full_text'])
            line_to_write = '{"source": "' + INSTRUCTION_NOTES + full_text + '"}\n' 
            jsonl_file.writelines([line_to_write])
        except Exception as e:
            # Print the line that caused the exception
            print('\\nError in {}'.format(row['head']))
            traceback.print_exc()
            import IPython; IPython. embed(); exit(1)
    jsonl_file.close()
    print(f"Generated {out_jsonl_path}")

def auto_evaluate_generations(in_tsv, out_tsv):
    df = pd.read_csv(in_tsv, sep='\t', header=0)

    # Check if manual evaluation is needed
    for index, row in df.iterrows():
        if pd.isnull(row['full_text']) and row['flagged_answer'] and not pd.isnull(row['generated_tail']) and pd.isnull(row['raw_answer']):
            raise Exception(f"*** Manual evaluation is needed for {row['head']} of {args.in_tsv}. Please check all with not filled results and fill the full_text column, adjust generated_tail, and set flagged_answer to False. Then run this script again.")
        
    # Placeholder for review columns (three annotators)
    df['review_1'], df['review_2'], df['review_3'], df['review'] = 0, 0, 0, 0
    # Iterate over rows and auto-evaluate
    for index, row in df.iterrows():
        auto_evaluated, selection = auto_evaluate_row(index, row)
        if auto_evaluated:
            df.at[index, 'review_1'] = selection
            df.at[index, 'review_2'] = selection
            df.at[index, 'review_3'] = selection
            df.at[index, 'review'] = selection
    # Saving the dataframe
    df.to_csv(out_tsv, sep='\t', index=False)

def fix_grammar(s):
    tool = language_tool_python.LanguageTool('en-US')
    is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and rule.context[rule.offsetInContext:rule.offsetInContext] in ['PersonX', 'PersonY']
    matches = tool.check(s)
    matches = [rule for rule in matches if not is_bad_rule(rule)]
    return language_tool_python.utils.correct(s, matches)

def easy_fix_grammar(s):
    s = s.replace('to to', 'to').replace('  ', ' ')
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tsv", type=str, default=f'./experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3.tsv')
    parser.add_argument("--action", type=str, choices=['auto_evaluate', 'generate_jsonl'], default='auto_evaluate')
    args = parser.parse_args()

    # Extracting file name
    filename = os.path.basename(args.in_tsv).split('.')[0]
    work_path = os.path.dirname(args.in_tsv)
    mturk_parent_path = f'{work_path}/mturk'
    if not os.path.exists(mturk_parent_path):
        os.makedirs(mturk_parent_path)
    out_jsonl_path = os.path.join(mturk_parent_path, f'{filename}_mturk.jsonl')
    out_tsv = os.path.join(work_path, f'{filename}_evaluated.tsv')

    # Auto-evaluating the generations with clear issues
    if args.action == 'auto_evaluate':
        auto_evaluate_generations(args.in_tsv, out_tsv)

    # Generating the jsonl file with the ones not auto-evaluated
    if args.action == 'generate_jsonl':
        generate_jsonl_for_mturk(out_tsv, out_jsonl_path)
    
    """ List of calls:
    - self_examples_eval:
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/self_samples_eval/few_shot_self_samples_to_eval_negated_preds_with_gpt_3.tsv
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/self_samples_eval/few_shot_self_samples_to_eval_preds_with_gpt_3.tsv
    - atomic_2020_eval:
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3.tsv
    python scripts/prepare_generations_for_mturk_evaluation.py --in_tsv experiments/atomic_2020_eval/few_shot_sampled_to_eval_with_gpt_3.tsv
    """