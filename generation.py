import random
from typing import Dict, List, Optional
import ollama

from vector_stores import (
    ProblemSolutionsStore,
    KarapavichusSolutionsStore,
    CBTSolutionsStore
)

def adapt_cbt_solution(query: str, cbt_matches: List[Dict], num_predict: int = 1000) -> Optional[str]:
    """
    Attempt to adapt a CBT solution to the user's specific problem.
    Returns None if no confident adaptation is possible.
    """
    if not cbt_matches:
        return None

    # Randomly select one CBT match
    match = random.choice(cbt_matches)

    prompt = f"""You are a CBT specialist tasked with adapting an existing CBT solution to a new problem.
    You should ONLY respond if you see a clear and meaningful way to adapt the solution.
    If you're not confident about the adaptation, respond with "NOT_APPLICABLE".

    Original Case:
    Problem: {match['thought']}
    CBT Solution: {match['cbt_plans'][0]}

    New Problem:
    {query}

    If you can confidently adapt this CBT approach to the new problem:
    1. Explain in 1-2 sentences how this CBT technique could be specifically applied
    2. Provide ONE concrete exercise based on this technique
    3. Start your response with "CBT_ADAPTATION:"

    If you're not confident about the adaptation, just respond with "NOT_APPLICABLE"

    Response:"""

    response = ollama.generate(
        model='llama3.2:latest',
        prompt=prompt,
        options={
            'temperature': 0.1,  # Low temperature for more focused adaptation
            'num_predict': num_predict,
        }
    )

    response_text = response.response.strip()
    if response_text.startswith("CBT_ADAPTATION:"):
        return response_text[15:].strip()  # Remove the prefix
    return None

def generate_response(query: str, store_results: Dict[str, List[Dict]], num_predict: int = 3000) -> str:
    """
    Generate personalized response based on matches from different stores.
    """
    # Extract relevant patterns
    problem_solutions = store_results.get('problem_solutions', [])
    karpavichus_matches = store_results.get('karpavichus_solutions', [])

    # Try to adapt a CBT solution
    cbt_adaptation = adapt_cbt_solution(query, store_results.get('cbt_solutions', []))

    prompt = f"""You are a professional counselor providing a ONE-TIME response to a person seeking help.
    This is not a chat conversation - you need to provide comprehensive yet concise guidance in a single response.

    USER'S SITUATION:
    {query}

    RELEVANT PATTERNS:
    {_format_patterns(problem_solutions)}

    THERAPEUTIC PRINCIPLES:
    {_format_principles(karpavichus_matches)}

    {_format_cbt(cbt_adaptation) if cbt_adaptation else ""}

    Provide a single, comprehensive response that:
    1. Shows understanding of their specific situation (1-2 sentences)
    2. Offers clear therapeutic insight (1-2 key points)
    3. Suggests concrete, actionable steps
    4. Acknowledges this is a one-time exchange

    Keep your response focused and concise. Do not:
    - Ask questions (this is not a chat)
    - Suggest seeking professional help (assume they know this option)
    - Provide general advice not connected to their specific situation
    - Use placeholder examples

    Response:"""

    response = ollama.generate(
        model='llama3.2:latest',
        prompt=prompt,
        options={
            'temperature': 0.7,
            'presence_penalty': 1.5,
            'num_predict': num_predict,
        }
    )

    return response.response

def _format_patterns(matches: List[Dict]) -> str:
    """Format most relevant patterns from similar situations."""
    if not matches:
        return "No direct matching patterns found."

    patterns = []
    for match in matches[:2]:  # Only use top 2 matches
        patterns.append(f"- In similar situations where {match['problem'].lower()}, "
                      f"successful approaches focused on {_extract_core(match['solution'])})")

    return "\n".join(patterns)

def _format_principles(matches: List[Dict]) -> str:
    """Format therapeutic principles from Karpavichus solutions."""
    if not matches:
        return "No specific therapeutic principles identified."

    principles = []
    for match in matches[:2]:  # Only use top 2 matches
        principles.append(f"- {match['solution']}")

    return "\n".join(principles)

def _format_cbt(adaptation: Optional[str]) -> str:
    """Format CBT adaptation if available."""
    if not adaptation:
        return ""

    return f"\nRELEVANT CBT APPROACH:\n{adaptation}"

def _extract_core(solution: str) -> str:
    """Extract core approach from solution text."""
    # Remove "The person should" and similar phrases
    core = solution.replace("The person should", "").strip()
    # Take first sentence and remove period
    core = core.split('.')[0].strip('.')
    return core.lower()

class ResponseGenerator:
    """Class to handle response generation using vector stores."""

    def __init__(self):
        self.stores = {
            'problem_solutions': ProblemSolutionsStore(),
            'karpavichus_solutions': KarapavichusSolutionsStore(),
            'cbt_solutions': CBTSolutionsStore()
        }

    def process_query(self, query: str) -> str:
        # Get matches from each store
        results = {}
        for name, store in self.stores.items():
            results[name] = store.search([query], top_k=3)

        # Generate response
        return generate_response(query, results)

# Example usage
def main():
    generator = ResponseGenerator()
    query = "I've been feeling overwhelmed at work lately. Whenever my boss assigns me new tasks, I freeze up and can't seem to prioritize or get started."
    response = generator.process_query(query)
    print(response)

if __name__ == "__main__":
    main()
