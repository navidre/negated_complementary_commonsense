
## Scripts order:
1. [Preparation](/scripts/prepare_subjects_preds_for_generation.py)
    - Input: The list of possible triples that we want to generate base on of, such as [ATOMIC-2020's test split](data/atomic2020/test.tsv).
    - Output: TSV file including rows sampled from each predicate, such as 10 triples per predicate. [A sample output file for the negated predicates](experiments/atomic_2020_eval/sampled_to_eval_negated_pred.tsv). The output has the verbalized verbalized version of subject & object as prompt column.
2. [Generation](/scripts/generate_objects_using_gpt_3.py)
    - Input: The sampled TSV file for evaluation from the preparation step.
    - Output: Generated tails (objects), given the subjects and predicates using an LLM, such as GPT-3. [Sample output file](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3.tsv).
3. [Self Human Evaluation](/scripts/human_evaluate_generations.py)
    - Input: TSV file with generations from the previous step.
    - Output: The script for self-human evaluation has two output files: one a TSV file including the human evaluations based on the categories ([sample](/experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated.tsv)) and a txt file including the accuracy and the category definitions ([sample](/experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated.txt)).
4. [Plotting](/scripts/plot_evaluated_results.py)
    - Input: The evaluations from the previous step.
    - Output: PDF file of plot ([sample](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted_results.pdf)) and the JSON file of aggregations for plotting ([sample](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted_results.json))