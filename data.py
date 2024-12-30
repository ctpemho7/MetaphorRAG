import pandas as pd
import numpy as np
import json
import tqdm
import os
from dataclasses import dataclass

import ollama
from datasets import load_dataset

# Path Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEXTS_DIR = os.path.join(BASE_DIR, 'texts')
KARPAVICHUS_TEXTS_DIR = os.path.join(TEXTS_DIR, 'karpavichus', 'split_en')

# Output file paths
MHC_EXTRACTIONS_PATH = os.path.join(DATA_DIR, 'MHC_extractions.json')
MHCC_EXTRACTIONS_PATH = os.path.join(DATA_DIR, 'MHCC_extractions.json')
CACTUS_PROCESSED_PATH = os.path.join(DATA_DIR, 'cactus_processed.json')
KARPAVICHUS_PROCESSED_PATH = os.path.join(DATA_DIR, 'karpavichus_processed.json')
KARPAVICHUS_PROBLEMS_PATH = os.path.join(DATA_DIR, 'karpavichus_problems.json')

FINAL_PATHS = {
    'problem_solutions': os.path.join(DATA_DIR, 'problem_solutions.json'),
    'karpavichus_solutions': os.path.join(DATA_DIR, 'karpavichus_solutions.json'),
    'cbt_solutions': os.path.join(DATA_DIR, 'cbt_solutions.json')
}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEXTS_DIR, exist_ok=True)


# Required Classes (add implementations)
class ExtractSolutionsFromMHC:
    prompt = """You are a specialized parser that extracts clear problem-solution pairs from counseling interactions. Your task is to:

    1. Extract distinct problems from the Questions, stating them as objective situations without personal pronouns
    2. Match these with solutions from the Answers, reformulating them in impersonal form starting with "The person should..."
    3. Ensure each solution is specific and actionable
    4. Present the output as an array of problem-solution pairs

    Guidelines:
    - Problems should be stated as situations/conditions, not personal statements
    - Solutions must begin with "The person should..." and be clear, actionable advice
    - Each problem should have at least one corresponding solution
    - Keep the language clear and professional

    Output only the JSON object with a field "pairs" that contains an array of problem-solution pairs.

    Example Input:
    {
        "Questions": "I'm fine when we start becoming intimate, but out of nowhere, I will get a flashback of what happened to me in the past. I start hysterically crying and freaking out when my boyfriend obviously has done nothing to hurt me.",
        "Answers": "Sexual intimacy can be very triggering for survivors even when it is both wanted and consensual. You may want to consider seeing a therapist who specializes in trauma to work through the abuse if you have not already done so. Often times triggers still hold such a powerful effect when the emotions related to the abuse  have not been fully processed. In the  meantime, you may want to consider coming up with a Safe Word to let your partner know that you are being triggered or to  communicate your physical boundaries to him. Often times, the experience of communicating  your physical boundaries to your partner, having those boundaries respected and validated, and having a partner who is understanding and  willing to engage in intimacy in such a way that does not violate your physical boundaries  can reinforce a sense of safety with him."
    }

    Example Output:
    {
        "pairs": [
            {
              "problem": "Flashbacks of past trauma during intimate moments",
              "solution": "The person should consult a therapist specializing in trauma to process the abuse-related emotions"
            },
            {
              "problem": "Sudden emotional distress during intimacy despite safe environment",
              "solution": "The person should establish a Safe Word system with their partner to communicate trigger moments"
            },
            {
              "problem": "Difficulty managing physical boundaries during intimate moments",
              "solution": "The person should clearly communicate physical boundaries to their partner and establish a system for respecting these boundaries during intimate moments"
            }
        ]
    }

    User Input:
    {
        "Questions": {questions},
        "Answers": {answers}
    }
    Assistant Output:"""

    schema = {
      "type": "object",
      "properties": {
        "pairs": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "problem": {
                "type": "string",
                "description": "A clear statement of a specific issue from the Context"
              },
              "solution": {
                "type": "string",
                "description": "A corresponding solution in impersonal form (the person should...)"
              }
            },
            "required": ["problem", "solution"],
            "additionalProperties": False
          }
        }
      },
      "required": ["pairs"],
      "additionalProperties": False
    }

    def format(self, row):
        return ExtractSolutionsFromMHC.prompt\
            .replace('{questions}', row['Questions'])\
            .replace('{answers}', row['Answers'])

@dataclass
class ExtractSolutionsFromMHCC:
    prompt = """You are a specialized parser that extracts clear problem-solution pairs from counseling interactions. Your task is to:

    1. Extract distinct problems from the Context, stating them as objective situations without personal pronouns
    2. Match these with solutions from the Response, reformulating them in impersonal form starting with "The person should..."
    3. Ensure each solution is specific and actionable
    4. Present the output as an array of problem-solution pairs

    Guidelines:
    - Problems should be stated as situations/conditions, not personal statements
    - Solutions must begin with "The person should..." and be clear, actionable advice
    - Each problem should have at least one corresponding solution
    - Keep the language clear and professional

    Output only the JSON object with a field "pairs" that contains an array of problem-solution pairs.

    Example Input:
    {
        "Context": "I'm fine when we start becoming intimate, but out of nowhere, I will get a flashback of what happened to me in the past. I start hysterically crying and freaking out when my boyfriend obviously has done nothing to hurt me.",
        "Response": "Sexual intimacy can be very triggering for survivors even when it is both wanted and consensual. You may want to consider seeing a therapist who specializes in trauma to work through the abuse if you have not already done so. Often times triggers still hold such a powerful effect when the emotions related to the abuse  have not been fully processed. In the  meantime, you may want to consider coming up with a Safe Word to let your partner know that you are being triggered or to  communicate your physical boundaries to him. Often times, the experience of communicating  your physical boundaries to your partner, having those boundaries respected and validated, and having a partner who is understanding and  willing to engage in intimacy in such a way that does not violate your physical boundaries  can reinforce a sense of safety with him."
    }

    Example Output:
    {
        "pairs": [
            {
              "problem": "Flashbacks of past trauma during intimate moments",
              "solution": "The person should consult a therapist specializing in trauma to process the abuse-related emotions"
            },
            {
              "problem": "Sudden emotional distress during intimacy despite safe environment",
              "solution": "The person should establish a Safe Word system with their partner to communicate trigger moments"
            },
            {
              "problem": "Difficulty managing physical boundaries during intimate moments",
              "solution": "The person should clearly communicate physical boundaries to their partner and establish a system for respecting these boundaries during intimate moments"
            }
        ]
    }

    User Input:
    {
        "Context": {context},
        "Response": {response}
    }
    Assistant Output:"""

    schema = {
      "type": "object",
      "properties": {
        "pairs": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "problem": {
                "type": "string",
                "description": "A clear statement of a specific issue from the Context"
              },
              "solution": {
                "type": "string",
                "description": "A corresponding solution in impersonal form (the person should...)"
              }
            },
            "required": ["problem", "solution"],
            "additionalProperties": False
          }
        }
      },
      "required": ["pairs"],
      "additionalProperties": False
    }

    def format(self, row):
        return ExtractSolutionsFromMHCC.prompt\
            .replace('{context}', row['Context'])\
            .replace('{response}', row['Response'])

@dataclass
class ExtractProblemsFromCactus:
    prompt = """You are a specialized parser that extracts clear problem statements from mental health-related thoughts. Your task is to:

    1. Extract distinct problems from the thought, stating them as objective situations without personal pronouns
    2. Break down complex thoughts into separate problem statements if multiple exist
    3. Present each problem as a clear, professional statement of a condition or situation

    Guidelines:
    - Problems should be stated as situations/conditions, not personal statements
    - Remove emotional language while preserving the core issue
    - Keep the language clear and professional
    - Each problem should be distinct and specific

    Output only the JSON object with a field "problems" that contains an array of problem statements.

    Example Input:
    {
        "thought": "I frequent this animal shelter. All of the animals remembered me except a few, I can never go back there again they will hate me"
    }

    Example Output:
    {
        "problems": [
            "Catastrophic thinking about minor social rejection",
            "Excessive emotional response to perceived animal indifference",
            "Self-imposed isolation due to fear of rejection"
        ]
    }

    User Input:
    {
        "thought": {thought}
    }
    Assistant Output:"""

    schema = {
      "type": "object",
      "properties": {
        "problems": {
          "type": "array",
          "items": {
            "type": "string",
            "description": "A clear statement of a specific issue from the thought"
          }
        }
      },
      "required": ["problems"],
      "additionalProperties": False
    }

    def format(self, row):
        return ExtractProblemsFromCactus.prompt\
            .replace('{thought}', row['thought'])

@dataclass
class ExtractPsychologicalInsights:
    prompt = """You are a specialized psychological assistant that processes text to identify key counseling insights. Your goal is to produce a concise JSON output that addresses three top issues found in the text. To accomplish this, follow the steps below:

    1. Brief Scenario Summary (Context)
       - Summarize the overall situation in no more than two sentences.
       - This summary should highlight the main people or parties involved, their conflict or situation, and any notable emotional dynamics.

    2. Identify Three Top Issues
       - Label each issue as “first”, “second”, and “third”.
       - Each issue should focus on a specific emotional or relational challenge, revealed by the text.
       - Use plain, conversational language that is clear and direct.

    3. Create Personalized Action Paths for Each Issue
       - For each issue, produce three pieces of information:

         a) Problem
            - A concise, fact-based statement about what the problem might be
            - Example: “The parents that are under a lot of stress, due to raising a child, frequently interrupt one another during discussions.”

         b) I-feel
            - How would this situation affect me ONLY psychologically if I was experiencing it?
            - A short personal or emotional angle as an I-statement, answering the question above.
            - Its reason should be sourced from the Problem, and it should reformulate the neccessary parts, because it neeeds to be understood without a context.
            - The subject should be "I", it should be first-person singular sentence.
            - Example: “I feel dismissed and stressed because I can’t get my full point across.”

         c) Practical Step
            - A single, specific, and actionable solution that directly addresses the issue.
            - It must be tangible and testable in everyday life without requiring professional intervention.
            - Example: “Agree on a 3-minute timer for each speaker. After one finishes, the listener repeats back what was said to confirm understanding before responding.”

    4. Final Output Requirements
       - Provide the result in JSON format.

    Example Input:
    {
        "text": "During our recent session, a client expressed frustration about feeling overshadowed in financial decisions. She and her partner frequently argued about budgeting—each believing their own approach was the only correct one. Neither side was willing to compromise or hear the other’s reasons, leading to constant tension."
    }

    Example Output:
    {
        "scenario_summary": "A couple struggles with budgeting disagreements, each partner insisting on their own financial approach without compromise.",
        "issues": [
            {
              "issue_label": "first",
              "problem": "They frequently interrupt and dismiss each other's budgeting suggestions.",
              "I-feel": "I feel invalidated and stressed because my ideas are brushed aside.",
              "practical_step": "Agree on a structured time to speak: each partner has 2 minutes to present a budget plan, and the other must repeat the plan back before proposing changes."
            },
            {
              "issue_label": "second",
              "problem": "They blame each other for financial mistakes instead of collaborating on solutions.",
              "I-feel": "I feel targeted and defensive, making discussions tense and unproductive.",
              "practical_step": "Start budget talks by listing shared financial goals. Use a whiteboard to write them down, so you both focus on solutions that serve these goals instead of assigning blame."
            },
            {
              "issue_label": "third",
              "problem": "They rarely accept any middle ground, insisting on one 'right' way to handle money.",
              "I-feel": "I feel anxious and worried about never finding a fair compromise.",
              "practical_step": "Experiment with a 30-day trial of each partner’s suggestion on a small scale (like groceries only). Collect results and review them together, focusing on which aspects worked or didn’t, to create a blended approach."
            }
        ]
    }

    User Input:
    {
        "text": {text}
    }

    Assistant Output:"""

    schema = {
      "type": "object",
      "properties": {
        "scenario_summary": {
          "type": "string",
          "description": "A brief, one-to-two-sentence overview of the situation."
        },
        "issues": {
          "type": "array",
          "description": "Array of three key issues identified in the scenario",
          "minItems": 3,
          "maxItems": 3,
          "items": {
            "type": "object",
            "properties": {
              "issue_label": {
                "type": "string",
                "enum": ["first", "second", "third"],
                "description": "Label indicating the order of the issue"
              },
              "problem": {
                "type": "string",
                "description": "Short, factual description."
              },
              "I-feel": {
                "type": "string",
                "description": "How does it affect me?"
              },
              "practical_step": {
                "type": "string",
                "description": "Actionable advice in one or two sentences."
              }
            },
            "required": ["issue_label", "problem", "I-feel", "practical_step"],
            "additionalProperties": False
          }
        }
      },
      "required": ["scenario_summary", "issues"],
      "additionalProperties": False
    }

    def format(self, row):
        return ExtractPsychologicalInsights.prompt\
            .replace('{text}', row['text'])

@dataclass
class ConvertToUserProblemAndSolution:
    prompt = """You are a specialized psychological assistant that converts abstract psychological insights into concrete problem-solution pairs.

        INPUT FORMAT:
        {
            "I-feel": "Abstract psychological/emotional experience or observation",
            "insight": "Abstract guidance or recommendation for improvement"
        }

        OUTPUT FORMAT:
        Generate {num_pairs} JSON objects, each containing:
        1. problem: A specific difficulty someone might seek counseling for
        2. insight: How the input insight applies to this specific problem
        3. solution: Concrete steps based on the insight

        GENERATION RULES:

        1. PROBLEMS MUST:
            - Use natural language a client would use when seeking help
            - Describe a specific situation or difficulty
            - Be 5-15 words long
            - Follow these patterns:
              * "Difficulty [doing X] because of [Y]"
              * "[Negative experience] when [specific situation]"
              * "[Behavioral pattern] leading to [negative outcome]"
              * "Fear of [specific thing] due to [specific reason]"
            Examples:
              * "Difficulty rebuilding trust after infidelity"
              * "Recurring nightmares about men trying to hurt the person"
              * "Struggling with loneliness when not in contact with partner"

        2. INSIGHTS MUST:
            - Begin with "In this case, [input insight] means..."
            - Explain how the abstract insight applies to the specific problem
            - Be 1-2 sentences long
            - Connect the general principle to the specific situation

        3. SOLUTIONS MUST:
            - Begin with "The person should..."
            - Contain exactly three sentences:
              1. Initial action step
              2. Follow-up practice
              3. Long-term habit formation
            - Be specific and immediately actionable
            - Directly apply the insight to the problem

        Example Input:
        {
            "I-feel": "The person feels a sense of belonging and acceptance within their social group when they share the same values and ideas as others. However, this can also lead to a lack of critical thinking and independence.",
            "insight": "The person needs to find ways to balance their desire for belonging and acceptance with their need for critical thinking and independence."
        }

        Example Output:
        [
            {
                "problem": "Fear of losing friends by expressing different opinions from the group",
                "insight": "In this case, balancing belonging with independence means learning to express disagreement while maintaining connections. This requires developing confidence in one's own views while showing respect for others.",
                "solution": "The person should start by expressing one small difference of opinion per week in low-stakes conversations with their most trusted friend. They should practice preparing thoughtful, respectful comments that begin with 'I see it differently because...' followed by their perspective. They should gradually expand this practice to larger group settings while maintaining a support journal of positive responses to their authentic self-expression."
            },
            {
                "problem": "Constantly changing personal views to match those of friends",
                "insight": "In this case, balancing belonging with independence means developing a stable sense of self while remaining open to genuine influence from others.",
                "solution": "The person should establish a '24-hour rule' of waiting one full day before adopting any new opinion expressed by friends. They should maintain a personal values journal documenting their authentic beliefs and the reasons behind them. They should practice phrases like 'I need time to think about that' when feeling pressured to immediately agree with others."
            }
        ]

        User Input:
        {
            "I-feel": {I_feel},
            "insight": {insight},
        }

        Assistant Output: Generate {num_pairs} problem-solution pairs following the exact format and rules above."""

    schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "A specific client problem statement"
                    },
                    "insight": {
                        "type": "string",
                        "description": "Application of the general insight to this specific problem"
                    },
                    "solution": {
                        "type": "string",
                        "description": "Three-part actionable solution starting with 'The person should'"
                    }
                },
                "required": ["problem", "insight", "solution"],
                "additionalProperties": False
            },
            "minItems": 1
        }

    def format(self, row):
        return ConvertToUserProblemAndSolution.prompt\
            .replace('{I_feel}', row['I-feel'])\
            .replace('{insight}', row['insight'])\
            .replace('{num_pairs}', str(row['num_pairs']))

def read_files_from_directory(directory_path):
    files_data = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # Check if file is a txt file
            file_path = os.path.join(directory_path, filename)

            # Read the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Create dictionary with filename and content
                files_data.append({
                    "filename": filename,
                    "text": content
                })
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    return files_data

def parse_with_ollama_from_class(row, class_type, num_predict=1000, t=0.01):
    try:
        prompt = class_type.format(row)
    except Exception as e:
        print(e)
        return None
    output = ollama.generate(
        model='llama3.2:latest',
        prompt=prompt,
        format=class_type.schema,
        options={
            'temperature': t,
            'presence_penalty': 2,
            'num_predict': num_predict,
            "penalize_newline": True
        }
    )
    try:
        return json.loads(output.response)
    except Exception as e:
        print(e)
        return output.response


def load_datasets():
    print("Loading Mental Health Counseling Conversations dataset...")
    dataset = load_dataset("Amod/mental_health_counseling_conversations")

    print("Loading Cactus dataset...")
    cactus_ds = load_dataset("DLI-Lab/cactus")

    print("Loading Mental Health Conversations dataset...")
    dataset_MHC = load_dataset("Kiran2004/MentalHealthConversations")

    return {
        'MHCC': dataset,
        'cactus': cactus_ds,
        'MHC': dataset_MHC
    }

def extract_valid_json_elements_robust(incomplete_json_str):
    """
    Combining multiple approaches with additional validation to extract viable json elements from a string.
    """
    import re

    def is_valid_json_object(s):
        try:
            parsed = json.loads(s)
            return isinstance(parsed, dict)
        except:
            return False

    def cleanup_json_object(s):
        # Remove trailing commas
        s = re.sub(r',\s*}', '}', s)
        # Add missing closing braces
        if s.count('{') > s.count('}'):
            s += '}' * (s.count('{') - s.count('}'))
        return s

    valid_elements = []

    # First try regex approach
    pattern = r'{[^{]*?}'
    potential_objects = re.finditer(pattern, incomplete_json_str)

    for match in potential_objects:
        obj_str = cleanup_json_object(match.group())
        if is_valid_json_object(obj_str):
            valid_elements.append(json.loads(obj_str))

    # If regex approach failed, try string splitting
    if not valid_elements:
        elements = incomplete_json_str.split('},')
        for element in elements:
            element = cleanup_json_object(element)
            if is_valid_json_object(element):
                valid_elements.append(json.loads(element))

    return valid_elements

def main():
    # Load datasets
    datasets = load_datasets()

    dataset_MHCC = datasets['MHCC']
    dataset_MHC = datasets['MHC']
    cactus_ds = datasets['cactus']

    # Process MHCC dataset
    MHCC_extractions = []
    for i in tqdm.tqdm(range(len(dataset_MHCC['train']))):
        MHCC_extractions.append(
            parse_with_ollama_from_class(dataset_MHCC['train'][i], ExtractSolutionsFromMHCC())
        )

    # Save MHC extractions
    with open(MHCC_EXTRACTIONS_PATH, 'w') as f:
        json.dump(MHCC_extractions, f)

    # Process MHC dataset
    MHC_extractions = []
    for i in tqdm.tqdm(range(len(dataset_MHC['train']))):
        MHC_extractions.append(
            parse_with_ollama_from_class(dataset_MHC['train'][i], ExtractSolutionsFromMHC())
        )

    # Save MHC extractions
    with open(MHC_EXTRACTIONS_PATH, 'w') as f:
        json.dump(MHC_extractions, f)

    # Process Cactus dataset
    cactus_df = pd.DataFrame(cactus_ds['train'])
    cactus_unique = cactus_df['thought'].unique()

    cactus_extractions = []
    for i, thought in tqdm.tqdm(enumerate(cactus_unique), total=len(cactus_unique)):
        cactus_extractions.append(
            parse_with_ollama_from_class({'thought': thought}, ExtractProblemsFromCactus())
        )

    cactus_processed = [{
        'thought': thought,
        'extraction': info
    } for info, thought in zip(cactus_extractions, cactus_unique)]

    # Save Cactus extractions
    with open(CACTUS_PROCESSED_PATH, 'w') as f:
        json.dump(cactus_processed, f)

    # Process text files
    files = read_files_from_directory(KARPAVICHUS_TEXTS_DIR)

    file_extractions = {}
    for file in tqdm.tqdm(files):
        file_extractions[file['filename']] = parse_with_ollama_from_class(
            file,
            ExtractPsychologicalInsights(),
            num_predict=3000,
        )

    with open(KARPAVICHUS_PROCESSED_PATH, 'w') as f:
        json.dump(file_extractions, f)

    # Process additional insights
    karpavichus_derived_problems = []
    for key in tqdm.tqdm(file_extractions):
        for i, issue in enumerate(
            file_extractions[key]['issues']
        ):
            row = {
                'I-feel': issue['I-feel'],
                "insight": issue['practical_step'],
                'num_pairs': 5
            }
            response = parse_with_ollama_from_class(
                row, ConvertToUserProblemAndSolution(),
                num_predict=1000,
                t=0.01
            )
            karpavichus_derived_problems.append(response)
    with open(KARPAVICHUS_PROBLEMS_PATH, 'w') as f:
        json.dump(karpavichus_derived_problems, f)


    # extracting only viable jsons and building datasets for later matching
    problem_solutions = []
    karpavichus_solutions = []
    cbt_solutions = []
    # MHCC
    for i, extraction in enumerate(MHCC_extractions):
        if extraction is None:
            continue
        elif type(extraction) == str:
            valid_elements = extract_valid_json_elements_robust(
                extraction
            )
        else:
            valid_elements = extraction['pairs']
        problem_solutions.extend([
            {
                **item,
                'meta': f'MHCC_extractions/{i}'
            }
            for item in valid_elements
        ])
    # MHC
    for i, extraction in enumerate(MHC_extractions):
        if extraction is None:
            continue
        elif type(extraction) == str:
            valid_elements = extract_valid_json_elements_robust(
                extraction
            )
        else:
            valid_elements = extraction['pairs']
        problem_solutions.extend([
            {
                **item,
                'meta': f'MHC_extractions/{i}'
            }
            for item in valid_elements
        ])
    # Cactus
    cactus_df = cactus_df.set_index('thought')
    for i, info in enumerate(cactus_processed):
        selection = cactus_df.loc[info['thought']]
        if type(selection) == pd.Series:
            selection = pd.DataFrame([selection])
        extraction = info['extraction']
        if extraction is None:
            continue
        elif type(extraction) == str:
            valid_elements = extract_valid_json_elements_robust(
                extraction
            )
        else:
            valid_elements = extraction['problems']
        for problem in valid_elements:
            cbt_solutions.append({
                'thought': info['thought'],
                'problem': problem,
                'cbt_plans': selection['cbt_plan'].values,
                'meta': f'cactus_processed/{i}',
            })
    # Karapavichus
    for i, extraction in enumerate(karpavichus_derived_problems):
        if extraction is None:
            continue
        elif type(extraction) == str:
            valid_elements = extract_valid_json_elements_robust(
                extraction
            )
        else:
            valid_elements = extraction
        karpavichus_solutions.extend([
            {
                'problem': item['problem'],
                'solution': item['solution'],
                'meta': f'karpavichus_derived_problems/{i}',
           } for item in valid_elements
        ])

    # deduplicating for clarity
    problem_solutions = {v['problem']: v for v in problem_solutions if 'problem' in v}
    karpavichus_solutions = {v['problem']: v for v in karpavichus_solutions}
    cbt_solutions = {v['problem']: v for v in cbt_solutions}

    with open(FINAL_PATHS['problem_solutions'], 'w') as f:
        json.dump(problem_solutions, f)
    with open(FINAL_PATHS['karpavichus_solutions'], 'w') as f:
        json.dump(karpavichus_solutions, f)
    with open(FINAL_PATHS['cbt_solutions'], 'w') as f:
        json.dump(cbt_solutions, f)

if __name__ == "__main__":
    main()
