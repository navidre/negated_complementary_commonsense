import os, requests, random, sys, re
from operator import itemgetter
sys.path.append('./')
from utils.gpt_3_utils import generate_few_shot_qa, SELF_EVALUATE_PROMPT, NORMAL_SELF_EVALUATE_PROMPT
from tqdm import tqdm
import pandas as pd

# Negated
OUR_METHOD_RESULT_PATH = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated.tsv'
target_file_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_self_post_processed.tsv'
accuracy_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_negated_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_self_post_processed.txt'
# Normal
NORMAL_OUR_METHOD_RESULT_PATH = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_normal_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated.tsv'
normal_target_file_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_normal_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_self_post_processed.tsv'
normal_accuracy_path = 'experiments/atomic2020_ten_preds/cot_qa_updated_neg_teach_var_temp/sampled_normal_preds_generated_cot_qa_updated_neg_teach_var_temp_evaluated_self_post_processed.txt'

def calculate_accuracy_based_on_majority_vote_and_self_process(merged_evaluation):
    total_count_majority, total_correct_majority, total_incorrect_majority, total_unfamiliar_majority = 0, 0, 0, 0
    if len(merged_evaluation) == 0:
        return 0
    # calculate the accuracy
    for index, row in merged_evaluation.iterrows():
        # Ignore row if 'self_is_correct' is False
        if not row['self_is_correct']:
            continue
        total_count_majority += 1
        if row['majority_vote'] == 1 or row['majority_vote'] == 2:
            total_correct_majority += 1
        elif row['majority_vote'] == 3 or row['majority_vote'] == 4:
            total_incorrect_majority += 1
        elif row['majority_vote'] == 5:
            total_unfamiliar_majority += 1
    accuracy = total_correct_majority*100.0/total_count_majority
    return accuracy

def sef_evaluate_results(results_path, target_file_path, accuracy_path, prompt):
    # Check if the results have already been self-evaluated
    if os.path.exists(target_file_path):
        print(f'Results already self-evaluated at {target_file_path}')
        return
    # Load the results
    df = pd.read_csv(results_path, sep='\t')
    # Add column called self_is_correct
    df['self_is_correct'] = True
    # Iterate over the results
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Self-evaluating'):
        # get the question and answer
        question, answer = row['prompt'].replace(' Name three.', ''), row['generated_tail']
        # Generate the question to ask from GPT-3
        evaluation_question = f'Question: {question} Answer: {answer}. Is this answer correct? Reply with yes or no.'
        # Ask GPT-3
        text_answer, response = generate_few_shot_qa(prompt, evaluation_question, max_tokens=10, temperature=0)
        # Update the answer
        if 'no' in text_answer:
            df.at[index, 'self_is_correct'] = False
    # Calculate the accuracy
    accuracy = calculate_accuracy_based_on_majority_vote_and_self_process(df)
    print(f'Accuracy for negated complementary case: {accuracy}')
    # Save the results
    df.to_csv(target_file_path, sep='\t', index=False)
    # Save the accuracy
    with open(accuracy_path, 'w') as f:
        f.write(f'Accuracy for negated complementary case: {accuracy}')
    print(f'Accuracy saved to {accuracy_path}') 

# Calculate the accuracy for negated
sef_evaluate_results(OUR_METHOD_RESULT_PATH, target_file_path, accuracy_path, SELF_EVALUATE_PROMPT)
# Calculate the accuracy for normal
sef_evaluate_results(NORMAL_OUR_METHOD_RESULT_PATH, normal_target_file_path, normal_accuracy_path, NORMAL_SELF_EVALUATE_PROMPT)