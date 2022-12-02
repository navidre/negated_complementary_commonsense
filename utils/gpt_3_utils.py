import os
from pathlib import Path
from turtle import st
from dotenv import load_dotenv
import openai

HOME = str(Path.home())

TEMPERATURE = 0.7 # t
FREQUENCY_PENALTY = 0 # fp 
PRESENCE_PENALTY = 0 # pp
MAX_TOKENS = 250 # mt
TOP_P = 1 # tp
ENGINE = 'text-davinci-002' # e

load_dotenv(f'{Path().resolve()}/.env')
openai.api_key = os.environ['OPENAI_API_KEY']

FEW_SHOT_PROMPT = "- PersonX accepts PersonY's invitation. As a result, PersonY does not feel sad.\n- You are not likely to find car in house. \n- Hammer cannot be used for typing.\n- PersonX cuts PersonX. PersonX will not be happy.\n- PersonX runs. Before that, it is not needed that he bikes."

def generate_zero_shot_using_gpt_3(prompt:str, temperature:float=TEMPERATURE, max_tokens:int=MAX_TOKENS, top_p:float=TOP_P, frequency_penalty:float=FREQUENCY_PENALTY, presence_penalty:float=PRESENCE_PENALTY, engine:str=ENGINE):
    """ Generate a zero-shot response using GPT-3.

    Args:
        prompt (str): The prompt to use with the model.
        temperature (float, optional): Temperature. Defaults to TEMPERATURE.
        max_tokens (int, optional): Number of max tokens generated in the output. Defaults to MAX_TOKENS.
        top_p (float, optional): Nucleus sampling. Top possible tokens with cumulative probability of at least top_p. Defaults to TOP_P.
        frequency_penalty (float, optional): frequency_penalty. Defaults to FREQUENCY_PENALTY.
        presence_penalty (float, optional): presence_penalty. Defaults to PRESENCE_PENALTY.
        engine (str, optional): Name of the GPT-3 engine to use. Defaults to ENGINE.

    Returns:
        (str, str): The generated text answer and the overall response.
    """
    start_sequence = " "
    prompt_as_input = f"{prompt}{start_sequence}"
    response = openai.Completion.create(
                model=engine,
                prompt=prompt_as_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=presence_penalty,
                presence_penalty=frequency_penalty,
                logprobs=20
                )
    text_answer = response['choices'][0]['text']
    return text_answer, response

def generate_few_shot_using_gpt_3(few_shot_prompt:str, premise:str, temperature:float=TEMPERATURE, max_tokens:int=MAX_TOKENS, top_p:float=TOP_P, frequency_penalty:float=FREQUENCY_PENALTY, presence_penalty:float=PRESENCE_PENALTY, engine:str=ENGINE):
    """ Generate a zero-shot response using GPT-3.

    Args:
        prompt (str): The few-shot example prompt to use with the model.
        premise (str): The premise to use with the model. Verbalized subject + predicate.
        temperature (float, optional): Temperature. Defaults to TEMPERATURE.
        max_tokens (int, optional): Number of max tokens generated in the output. Defaults to MAX_TOKENS.
        top_p (float, optional): Nucleus sampling. Top possible tokens with cumulative probability of at least top_p. Defaults to TOP_P.
        frequency_penalty (float, optional): frequency_penalty. Defaults to FREQUENCY_PENALTY.
        presence_penalty (float, optional): presence_penalty. Defaults to PRESENCE_PENALTY.
        engine (str, optional): Name of the GPT-3 engine to use. Defaults to ENGINE.

    Returns:
        (str, str): The generated text answer and the overall response.
    """
    start_sequence = "- "
    prompt_as_input = f"{few_shot_prompt}\n{start_sequence}{premise}"
    response = openai.Completion.create(
                model=engine,
                prompt=prompt_as_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=presence_penalty,
                presence_penalty=frequency_penalty,
                logprobs=20,
                stop=[". ", "\n"]
                )
    text_answer = response['choices'][0]['text']
    return text_answer, response

# TODO: Update this for the CSQA dataset
def q_and_a_gpt3(prompt:str, question:str, temperature:float=TEMPERATURE, max_tokens:int=MAX_TOKENS, top_p:float=TOP_P, frequency_penalty:float=FREQUENCY_PENALTY, presence_penalty:float=PRESENCE_PENALTY, engine:str=ENGINE):
    """ Generic method for generating a response to a question. Q/A style of prompting.

    Args:
        prompt (str): The static part of the prompt to use with the model.
        question (str): The question to ask the model. Technically part of prompt.
        temperature (float, optional): Temperature. Defaults to TEMPERATURE.
        max_tokens (int, optional): Number of max tokens generated in the output. Defaults to MAX_TOKENS.
        top_p (float, optional): Nucleus sampling. Top possible tokens with cumulative probability of at least top_p. Defaults to TOP_P.
        frequency_penalty (float, optional): frequency_penalty. Defaults to FREQUENCY_PENALTY.
        presence_penalty (float, optional): presence_penalty. Defaults to PRESENCE_PENALTY.
        engine (str, optional): Name of the GPT-3 engine to use. Defaults to ENGINE.

    Returns:
        (str, str): The generated text answer and the overall response.
    """
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    prompt_as_input = f"{prompt}{restart_sequence}{question}{start_sequence}"
    response = openai.Completion.create(
                engine=engine,
                prompt=prompt_as_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=presence_penalty,
                presence_penalty=frequency_penalty,
                logprobs=20,
                stop=["\n"]
                )
    text_answer = response['choices'][0]['text']
    return text_answer, response