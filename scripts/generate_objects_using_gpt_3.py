import argparse, sys, os, traceback
import pandas as pd
from tqdm import tqdm
sys.path.append('./')
from utils.gpt_3_utils import generate_zero_shot_using_gpt_3, generate_few_shot_using_gpt_3, PROMPTS, generate_few_shot_qa, QUESTION_TEMPLATES, NUMBER_TO_TEXT

def extract_answers(text_answer, style, num_generations):
    """Extracts the answers from the text_answer

    Args:
        text_answer (str): The text_answer from the GPT-3 API
        style (str): The style of generation
        num_generations (int): The number of generations

    Returns:
        answers (list): The list of answers
    """
    # Checking obvious cases
    if text_answer is None:
        return None
    if text_answer == "":
        return None

    # Extracting answers based on the style
    answers = []
    if style == "few_shot_qa":
        answers = text_answer.split(';')
    elif style in ["cot_qa", "updated_cot_qa", "cot_qa_neg_teach", "cot_qa_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp_ablated"]:
        start_phrase = 'The answers are: '
        if start_phrase in text_answer:
            answers = text_answer.split(start_phrase)[-1].split(';')
            answers[-1] = answers[-1].replace('.', '')
        else:
            # Another try to extract answers
            answers = text_answer.split(';')
            answers[0] = answers[0].split(':')[1] if ':' in answers[0] else answers[0]
            answers[-1] = answers[-1].replace('.', '')
    else:
        raise NotImplementedError
    
    if len(answers) != num_generations:
        return None
    else:
        answers = [answer.strip() for answer in answers]
        return answers

if __name__ == "__main__":
    print('Loading spacy ...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv")
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--style", type=str, default="few_shot", choices=["few_shot", "cot_qa", "few_shot_qa", "updated_cot_qa", "cot_qa_neg_teach", "cot_qa_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp_ablated"])
    parser.add_argument("--negated", action="store_true")
    args = parser.parse_args()

    #TODO: Make sure we do not overwrite the existing generations (if generated file exists) 
    #TODO: and we do not re-do generation for the same prompt (if the prompt is already in the generated file)

    """Sample calls:
    - For normal (non-negated), few-shot generation:
    python scripts/generate_objects_using_gpt_3.py --input experiments/atomic_2020_eval/sampled_to_eval.tsv --style few_shot
    - For negated, few-shot generation:
    python scripts/generate_objects_using_gpt_3.py --input experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv --style few_shot --negated
    """

    # Extracting file name and defining the out file name and path
    filename = os.path.basename(args.input).split('.')[0]
    out_filename = f'{filename}_generated_{args.style}'
    work_path = os.path.dirname(args.input)
    out_tsv = os.path.join(work_path, f'{out_filename}.tsv')
    # Load args.input as a pandas dataframe and ignore the first row
    df = pd.read_csv(args.input, sep="\t", header=0)
    # Remove 'tail' column
    df = df.drop(columns=['tail'])
    # Add column 'flagged_answer' to df with value 'False'
    df['flagged_answer'] = False
    # Add column 'raw_answer' to df with value ''
    df['raw_answer'] = ''
    # Placeholder dataframe to store generated rows, which has args.num_generations rows for each row in df
    generated_df = pd.DataFrame(columns=['head', 'relation', 'prompt', 'generated_tail', 'full_text', 'flagged_answer', 'raw_answer'])
    # Negation string
    negation_str = 'negated' if args.negated else 'normal'

    # Iterate over all rows in df
    progress_bar = tqdm(df.iterrows())
    for index, row in progress_bar:
        progress_bar.set_description("Processing {}-{}".format(row['head'], row['relation']))
        # Exception handling if there is an error in the following lines
        try:
            # Make a copy of row and store in row_copy
            row_copy = row.copy()
            # Add 'generated_tail' and 'full_text' columns to row_copy
            row_copy['generated_tail'] = ''
            row_copy['full_text'] = ''
            # Generate args.num_generations rows for each row in df

            # Classify the process into one generation per run or multiple generations per run
            if args.style == "zero_shot" or args.style == "few_shot":
                # Styles with one row generation each time
                for i in range(args.num_generations):
                    if args.style == "zero_shot":
                        generated_tail, response = generate_zero_shot_using_gpt_3(row_copy['prompt'], max_tokens=20)
                    elif args.style == "few_shot":
                        generated_tail, response = generate_few_shot_using_gpt_3(PROMPTS[args.style][negation_str], row_copy['prompt'], max_tokens=20)
                    # Assigning generations to row_copy
                    row_copy['generated_tail'] = generated_tail.replace('\n', ' ').strip()
                    row_copy['full_text'] = f"{row_copy['prompt']} {row_copy['generated_tail']}".strip()
                    # Append the row_copy to generated_df
                    generated_df = generated_df.append(row_copy, ignore_index=True)
            else:
                # Styles with multiple rows generation each time
                if args.style in ["few_shot_qa", "cot_qa", "updated_cot_qa", "cot_qa_neg_teach", "cot_qa_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp_ablated"]:
                    # Either should be out of loop or should not generate the next times and only first time!
                    normal_relation = row_copy['relation'] if args.negated is False else row_copy['relation'][3:]
                    question = QUESTION_TEMPLATES[normal_relation][negation_str]
                    subj, n = row_copy['head'], NUMBER_TO_TEXT[args.num_generations]
                    question_str = eval('f' + repr(question))
                    old_prompt, row_copy['prompt'] = row_copy['prompt'], question_str
                    if args.style in ["cot_qa_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp", "cot_qa_updated_neg_teach_var_temp_ablated"]:
                        for temperature in [0.7, 1]:
                            text_answer, response = generate_few_shot_qa(PROMPTS[args.style][negation_str], question_str, max_tokens=150, temperature=temperature)
                            if text_answer != '':
                                break
                    else:
                        text_answer, response = generate_few_shot_qa(PROMPTS[args.style][negation_str], question_str, max_tokens=100)
                    # Extracting the answers from the text_answer
                    answers = extract_answers(text_answer, args.style, args.num_generations)
                    if answers is None:
                        row_copy['flagged_answer'] = True
                    # Adding answers to the generated_df
                    for i in range(args.num_generations):
                        row_copy['raw_answer'] = text_answer
                        if answers is not None:
                            answer = answers[i]
                            generated_tail = answer.replace('\n', ' ').strip()
                            full_text = f"{old_prompt} {generated_tail}".strip()
                        else:
                            generated_tail, full_text = text_answer, ''
                        # Assigning generations to row_copy
                        row_copy['generated_tail'] = generated_tail
                        row_copy['full_text'] = full_text
                        # Append the row_copy to generated_df
                        generated_df = generated_df.append(row_copy, ignore_index=True)
                else:
                    raise NotImplementedError
        except:
            # Print the line that caused the exception
            print('\nError in {}'.format(row['head']))
            traceback.print_exc()
            import IPython; IPython. embed(); exit(1)

        # Save generated_df as a tsv file in every step
        generated_df.to_csv(out_tsv, sep='\t', index=False)