
## Scripts order:
1. [Preparation](/scripts/prepare_subjects_preds_for_generation.py)
        - Input: The list of possible triples that we want to generate base on of, such as [ATOMIC-2020's test split](data/atomic2020/test.tsv).
        - Output: TSV file including rows sampled from each predicate, such as 10 triples per predicate. [A sample output file for the negated predicates](experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv). The output has the verbalized verbalized version of subject & object as prompt column.
2. [Generation](/scripts/generate_objects_using_gpt_3.py)
        - Input: The sampled TSV file for evaluation from the preparation step.
        - Output: Generated tails (objects), given the subjects and predicates using an LLM, such as GPT-3. [Sample output file](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3.tsv).
3. Evaluation:
We have an option here. Either do self-evaluation or use AWS mTurk for evaluation.
    
    a. [Self Human Evaluation](/scripts/human_evaluate_generations.py)
        - Input: TSV file with generations from the previous step.
        - Output: The script for self-human evaluation has two output files: one a TSV file including the human evaluations based on the categories ([sample](/experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated.tsv)) and a txt file including the accuracy and the category definitions ([sample](/experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated.txt)).

    b. AWS mTurk needs pre-processing to have the correct JSONL format, do the evaluations on AWS, and then post-processing to put it back in our own flow.
    
    1) [AWS mTurk Input File Preparation Script](prepare_generations_for_mturk_evaluation.py)
        - Input: TSV file with generations from the previous step.
        - Outputs: 
            1. Auto-evaluation first and generating evaluation file: [Example](experiments/self_samples_eval/few_shot_self_samples_to_eval_negated_preds_with_gpt_3_evaluated.tsv).
            2. JSONL file for mTurk evaluation: [Example](experiments/self_samples_eval/few_shot_self_samples_to_eval_negated_preds_with_gpt_3_mturk.jsonl).
    2) Evaluation on AWS mTurk.
        - Input: JSONL file from the last step.
        - Output: Manifest file from AWS mTurk.
    3) [Script to post-process the evaluation files]()
        - Input: Manifest file from AWS mTurk.
        - Output: TSV files with decisions and alpha score to be fed to the plotting script.

4. [Plotting](/scripts/plot_evaluated_results.py)
        - Input: The evaluations from the previous step.
        - Output: PDF file of plot ([sample](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted_results.pdf)) and the JSON file of aggregations for plotting ([sample](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted_results.json))