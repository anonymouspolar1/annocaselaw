### Task 1

INPUT_A = "These are the facts of the case:\n{facts}\n\nAnd this is the procedural history:\n{procedural_history}"
INPUT_B = INPUT_A + "\n\nAnd these are the relevant precedents:\n{relevant_precedents}"
INPUT_C = INPUT_B + "\n\nAnd this is how the law was applied to the facts:\n{application_of_law_to_facts}"

input_mapping = {
    'a': INPUT_A,
    'b': INPUT_B,
    'c': INPUT_C
    }

PROMPT = """You are a seasoned judge specializing in U.S. civil negligence appeals cases. Analyze the following case details and determine the appropriate appellate outcome: "affirm," "reverse," or "mixed" (if part of the decision is affirmed and part reversed).

{input}

Reason step by step and be as thorough as possible.

Respond in strict JSON format with a single key-value pair:

```json
{{"outcome": "affirm" | "reverse" | "mixed"}}

Provide no additional text or explanation."""

### Task 3

import typing

instructions_zeroshot = """
    Your job is to annotate the text in law case files with the annotation types:
    1. Facts
    2. Procedural History
    3. Application of Law to Facts
    4. Relevant Precedents
    5. Outcome

    First, clearly define the annotation types so you know what you are looking for.
    Then, taking your time and being as thorough as possible, work slowly through the law case file
    and decide if sections of text belong to one of the annotation types.
    Use one pass through the file for each annotation type.
    Combine all the annotations from a single annotation type into a list of comma separated strings.
    Annotations must be direct, unedited quotes from the case file.

    Important: Do not return sources; always use the response function to respond to the user,
    ensuring all fields are populated.
"""

message_zeroshot = """
    Annotate the attached .txt law case file as per your instructions.
    Important: Do not return sources; always use the response function to respond to the user,
    ensuring all fields are populated.
"""

annotation_types = {
    "Facts": "The facts in a negligence law case describe the specific actions, events, or circumstances that led to the plaintiff’s injury or harm, detailing who did what, when, and how the injury occurred.",
    "Procedural History": "Procedural history refers to the timeline and actions taken in the case, including decisions made by lower courts and the steps that led to the case being heard.",
    "Application of Law to Facts": "Application of law to facts involves analyzing the established facts in the case under the legal principles of negligence, such as duty, breach, causation, and damages, to determine whether the defendant is legally liable.",
    "Relevant Precedents": "Relevant precedents are prior decisions from higher courts which establish legal principles that guide the court's analysis of the negligence issues in the current case.",
    "Outcome": "The outcome is the court's final decision, either affirming, reversing, or modifying the lower court’s ruling, and determining whether the defendant is liable for negligence and if damages should be awarded to the plaintiff.",
}

response_schema = {
    "Facts": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are facts"
        },
    "Procedural History": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are procedural history"
    },
    "Application of Law to Facts": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are application of law to facts"
    },
    "Relevant Precedents": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are relevant precedents"
    },
    "Outcome": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are outcomes"
    },
}

alt_response_schema = {
    "Facts": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are facts"
        },
    "Procedural History": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are procedural history"
    },
    "Application of Law to Facts": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are application of law to facts"
    },
    "Relevant Precedents": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are relevant precedents"
    },
    "Outcome": {
        "type": "array",
        "items": {"type": "string"},
        "description": "The sections of text in the attached law case file that are outcomes"
    },
}

response_function = {
    "name": "Response",
    "description": "The structure all responses to the user should follow.",
    "parameters": {
        "type": "object",
        "required": [
            "Facts",
            "Procedural History",
            "Application of Law to Facts",
            "Relevant Precedents",
            "Outcome"
        ],
        "properties":  response_schema
    },
}



class OutputSchema(typing.TypedDict):
    Facts: list[str]
    Procedural_History: list[str]
    Application_of_Law_to_Facts: list[str]
    Relevant_Precedents: list[str]
    Outcome: list[str]