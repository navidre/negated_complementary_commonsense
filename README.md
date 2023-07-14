
## Paper
This code is used in the following paper accepted at Natural Language Reasoning and Structured Explanations (NLRSE) workshop co-located at ACL 2023:

Negated Complementary Commonsense using Large Language Models

Citation:
```
@misc{rezaei2023negated,
      title={Negated Complementary Commonsense using Large Language Models}, 
      author={Navid Rezaei and Marek Z. Reformat},
      year={2023},
      eprint={2307.06794},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Scripts order
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
    2) Evaluation on AWS mTurk. Further explained in the following AWS mTurk Evaluation section.
        - Input: JSONL file from the last step alongside evaluation instructions.
        - Output: Manifest file from AWS mTurk.
    3) [Script to post-process the evaluation files](scripts/post_process_mturk_evaluations.py)
        - Input: Manifest file from AWS mTurk. To download the files, you can use this command:
        ```
        aws s3 cp --recursive s3://[S3 Bucket]/[FOLDER]/ ./experiments/self_samples_eval/mturk_results
        ```
        (Folders can be changed based on the experiments.)
        - Output: TSV files with decisions and alpha score to be fed to the plotting script. Using the out_tsv from step 1.

4. [Plotting](/scripts/plot_evaluated_results.py)
        - Input: The evaluations from the previous step.
        - Output: PDF file of plot ([sample](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted_results.pdf)) and the JSON file of aggregations for plotting ([sample](experiments/atomic_2020_eval/few_shot_sampled_to_eval_negated_pred_with_gpt_3_self_evaluated_adjusted_results.json))

## Run pipeline
The following scripts make the multiple steps in the previous section easier. The overall process is preparation/generation, human-evaluation, and lastly post-processing/plotting.

1. Preparation/Generation/Upload-to-S3: [Simple Pre-Process and Generation](scripts/simple_pre_process_and_generate.py)
2. Human Evaluation. Here are the options:
    1. Using AWS SageMaker.
    2. Self-evaluation using the [Human-Evaluation Script](scripts/human_evaluate_generations.py).
3. Post-Processing/Plotting: [Simple Post Process](scripts/simple_post_process.py)
    NOTE: Please look at instructions to mention if downloading results from S3 or feeding locally evaluated file.

After doing all experiments, we can use [the comparison script](scripts/compare_methods.py) to compare all the methods in one table.

## AWS mTurk Evaluation

For evaluation, we used Amazon mTurk through the [AWS SageMaker's Ground Truth module](https://aws.amazon.com/sagemaker/data-labeling/). We created a labeling job with the following specifications and procedures: 

1) Created a specific bucket under S3 and a specific folder underneath for each experiment. Placed the JSONL file under this folder. An empty subdirectory created for the evaluation results as well.

2) Starting a new labeling job with the following specifications:
        
        - Manual data setup
        - "Input dataset location" pointing to the JSONL file and "Output dataset location" to the results subfolder, both from the last step.
        - Task: Text Classification (Single Label)
        - Worker types: Amazon Mechanical Turk
        - Appropriate timeout and task expiration time
        - Uncheck automated data labeling
        - Under additional configuration, select 3 workers.
        - Brief description:
            Based on your own commonsense, choose one of the five options. Examples are provided in the description.

            - Only for negated cases: 

            IMPORTANT: Please note the CANNOT, DO Not, and other negated cases.

        - Instructions:

            Unfamiliar to me to judge 

            - Example: PersonX discovers a new planet. The planet is in the Alpha Centauri system.

            First part and second part are not related! Or not enough information to judge.
            
            Example: PersonX rides a bike. Elephants are not birds.
            (Although second part is correct, it is not related to the first part)

            Makes sense:

            Example: It is NOT likely to see elephant on table.

            Does not make sense:

            Example: It is likely to see elephant on table.

        - Options:
            - Makes sense
            - Sometimes makes sense
            - Does not make sense or Incorrect
            - First part and second part are not related! Or not enough information to judge
            - Unfamiliar to me to judge
