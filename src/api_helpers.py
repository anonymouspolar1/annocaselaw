import os
import json
import numpy as np
import pandas as pd
import requests
from typing import Literal
import time

from openai import OpenAI
import google.generativeai as genai
from google.generativeai import GenerativeModel, configure

from src.prompts import *
from src.eval_helpers import amalgamate_annotations, evaluate

### Task 1

TEMPERATURE = 0.01
TOP_P = 0.9
REASONING_EFFORT = 'high'
MAX_ATTEMPTS = 5

MODEL_PRICING = {
    'gpt-4o-2024-11-20': {'input': 2.50, 'output': 10.0},
    'o3-mini-2025-01-31': {'input': 1.10, 'output': 4.40},
    'o1-2024-12-17': {'input': 15.0, 'output': 60.00},
    'models/gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
    'deepseek/deepseek-r1': {'input': 8.00, 'output': 8.00},
    }

class JudgmentPredictionAPI:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    def generate_prompt(self, case_file_path: str, subtask: str = Literal["a", "b", "c"]):
        with open(case_file_path, encoding='utf-8') as f:
            case_file = json.load(f)
        
        annotations = case_file['annotations']
        input = input_mapping.get(subtask).format(
            facts=annotations['Facts'],
            procedural_history=annotations['Procedural History'],
            relevant_precedents=annotations['Relevant Precedents'],
            application_of_law_to_facts=annotations['Application of Law to Facts']
            )
        
        prompt = PROMPT.format(input=input)
        return prompt

    
    def get_outcome_openai(self, model: str, prompt: str):
        params = {}
        if model in ['gpt-4o-2024-11-20']:
            params = {'temperature': TEMPERATURE,
                      'top_p': TOP_P}
            
        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            **params,
            )

        outcome = json.loads(completion.choices[0].message.content)['outcome']
        return outcome
    
    
    def get_outcome_google(self, model_name: str, prompt: str):
        model = GenerativeModel(model_name=model_name,
                                generation_config=genai.GenerationConfig(temperature=0.01, top_p=0.9))
        response = model.generate_content(prompt)

        outcome = json.loads(response.text.replace("json", "").replace("```", ""))['outcome']
        return outcome
    
    def get_outcome_openrouter(self, model: str, prompt: str):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            },
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "provider": {"sort": "throughput"},
            })
        )

        outcome = json.loads(response.json()['choices'][0]['message']['content'].replace("json", "").replace("```", ""))['outcome']
        return outcome
    

    def get_outcome(self, case_file_path, subtask, model):
        attempts = 0
        while attempts <= MAX_ATTEMPTS:
            try:
                prompt = self.generate_prompt(case_file_path, subtask)

                if model in ['gpt-4o-2024-11-20', 'o3-mini-2025-01-31', 'o1-2024-12-17']:
                    return self.get_outcome_openai(model, prompt)
                elif model in ['gemini-1.5-pro']:
                    return self.get_outcome_google(model, prompt)
                elif model in ['deepseek/deepseek-r1']:
                    return self.get_outcome_openrouter(model, prompt)
                
            except Exception as e:
                if attempts < MAX_ATTEMPTS:
                    attempts += 1
                    print("Retrying")
                    continue
                else:
                    raise


### Task 3

class AnnotatorOpenAI:
    """
    A class to analyze legal case files using OpenAI's Assistants API.
    """
    def __init__(self, model: str):
        """Initialize Annotator with an OpenAI assistant and a thread ready to annotate

        Args:
            model (str, optional): The valid OpenAI model to use.
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model

    def annotate_case(self, case_file_path: str, annotation_types: str):
        """Add case and associated message to thread, and get annotations"""
        with open(case_file_path, 'r', encoding='utf-8') as file:
            case_text = file.read()
        
        prompt = f"""Annotate the below law case file according to the following 5 annotation types:
        {json.dumps(annotation_types, indent=2)}
        
        Take your time, be as thorough as possible, and combine all the annotations from a single annotation type into a list of comma-separated strings. Do not include sources. Annotations must be direct, unedited quotes from the case file.

        Always respond to the user in JSON format where the keys are the annotation types, and the value for each key is an array (list) of strings where each string is a separate annotation relevant to the given key. Include no other text in your response.

        Law case file to annotate:
        {case_text}
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )

        preds = json.loads(completion.choices[0].message.content)
        return preds

class AnnotatorGoogle:
    def __init__(self, model, annotation_types: str, response_function: str):
        configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = GenerativeModel(model_name=model,
                                     generation_config=genai.GenerationConfig(temperature=0.0,))
        self.annotation_types = annotation_types
        self.response_function = response_function
        
    def annotate_case(self, case_file_path: str):
        """Annotate Case File"""
        with open(case_file_path, 'r', encoding='utf-8') as file:
            case_text = file.read()

        prompt = f"""Annotate the below law case file according to the following 5 annotation types:
        {json.dumps(self.annotation_types, indent=2)}
        
        Take your time, be as thorough as possible, and combine all the annotations from a single annotation type into a list of
        comma-separated strings. Do not include sources. Annotations must be direct, unedited quotes from the case file.
        None of the case file are copyrighted, and you can quote them freely in your annotations.

        Do not include sources, and always respond with the exact following JSON schema, and no other text:
        {json.dumps(self.response_function, indent=2)}.

        Legal text to annotate:
        {case_text}
        """

        response = self.model.generate_content(prompt)
        preds = json.loads(response.text.replace("json", "").replace("```", ""))
        return preds
    
def annotate_openai(case_file_path: str,
                    model: str = 'gpt-4o',
                    annotation_types: str = annotation_types,
                    max_attempts: int = 5):
    """Function to annotate a case, and score the predictions (with error handling)

    Args:
        case_file_path (str): Path to case file to be annotated
        model (str, optional): OpenAI model. Defaults to 'gpt-4o'.
        annotation_types (dict, optional): Details of how to annotate. Defaults to annotation_types
        max_attempts (int, optional): max_attempts if error encountered. Defaults to 5.

    Returns:
        dict: predicted annotations
        np.ndarray: score of predictions
    """
    for attempt in range(max_attempts):
        try:
            annotator = AnnotatorOpenAI(model)
            preds = annotator.annotate_case(case_file_path, annotation_types)

            ground_truth_path = case_file_path.replace("txt", "json")
            gt = amalgamate_annotations(ground_truth_path)
            score = evaluate(gt, preds)

            return preds, score

        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"\nAttempt {attempt + 1} failed: {str(e)}")
                time.sleep(5)
                continue
            else:
                print(f"\nAll {max_attempts} attempts failed")
                raise  # Re-raise the last exception

def annotate_google(case_file_path: str,
                    model: str = 'gemini-1.5-pro',
                    annotation_types: str = annotation_types,
                    response_function: str = response_schema,
                    max_attempts: int = 5):
    """Function to annotate a case, and score the predictions (with error handling)

    Args:
        case_file_path (str): Path to case file to be annotated
        model (str, optional): OpenAI model. Defaults to 'gpt-4o'.
        annotation_types (str, optional): The types of annotations that is used in the prompt
        response_function (str, optional): The JSON schema that responses to the user must follow. Defaults to response_schema
        max_attempts (int, optional): max_attempts if error encountered. Defaults to 5.

    Returns:
        dict: predicted annotations
        np.ndarray: score of predictions
    """
    for attempt in range(max_attempts):
        try:
            annoatator = AnnotatorGoogle(model, annotation_types, response_function)
            preds = annoatator.annotate_case(case_file_path)

            ground_truth_path = case_file_path.replace("txt", "json")
            gt = amalgamate_annotations(ground_truth_path)
            score = evaluate(gt, preds)

            return preds, score

        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"\nAttempt {attempt + 1} failed: {str(e)}")
                time.sleep(2)
                continue
            else:
                print(f"\nAll {max_attempts} attempts failed")
                raise


### Misc Helpers for Task 3

def clear_openai_cache():
    """Deletes all existing assistants and vector stores"""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    while len(client.beta.vector_stores.list(limit=100).data) > 0:
        for vs in client.beta.vector_stores.list(limit=100).data:
            client.beta.vector_stores.delete(vector_store_id=vs.id)

    while len(client.beta.assistants.list(limit=100).data) > 0:
        for i in client.beta.assistants.list(limit=100).data:
            client.beta.assistants.delete(assistant_id=i.id)

    print("Cache Clear")


def sample_cases(gt_folder, n_train: int, n_val: int, n_test: int, seed=1):
    """Sample good cases to train and test on"""
    # Ignore cases that have an empty annotation field
    bad_idxs = set()
    files = os.listdir(gt_folder)
    for i, filename in enumerate(files):
            file_path = os.path.join(gt_folder, filename)
            gt = amalgamate_annotations(file_path)
            if any([not v for k, v in gt.items()]):
                bad_idxs.add(i)

    np.random.seed(seed)
    available_idxs = list(set(range(len(files))) - bad_idxs)
    training_idxs = np.random.choice(available_idxs, n_train, replace=False)

    possible_val_idxs = list(set(available_idxs) - set(training_idxs))
    val_idxs = np.random.choice(possible_val_idxs, n_val, replace=False)

    possible_test_idxs = list(set(possible_val_idxs) - set(training_idxs))
    test_idxs = np.random.choice(possible_test_idxs, n_test, replace=False)

    training_data_paths = [os.path.join(gt_folder, files[i]) for i in training_idxs]
    validation_data_paths = [os.path.join(gt_folder, files[i]) for i in val_idxs]
    test_gtpaths = [os.path.join(gt_folder, files[i]) for i in test_idxs]

    return training_data_paths, validation_data_paths, test_gtpaths
    