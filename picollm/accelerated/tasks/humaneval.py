import re
from datasets import load_dataset
from picollm.accelerated.execution import execute_code
from picollm.accelerated.tasks.common import Task

def extract_imports(prompt):
    """Extract import statements from the beginning of a code block."""
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            break
    return '\n'.join(imports)

def extract_program(completion):
    """
    Extract Python code from LLM completion.

    Handles various output formats:
    - Code wrapped in ```python ... ``` or ``` ... ``` blocks
    - Plain code without markdown blocks
    - Extra text before/after code blocks

    Returns the first code block if found, otherwise returns the whole completion.
    """
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        return matches[0].strip()

    return completion.strip()

class HumanEval(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        prompt = row['prompt'] # prompts in HumanEval are the beginning of the program
        solution = row['canonical_solution'] # the correct continuation of the program
        entry_point = row['entry_point'] # the function to check
        test = row['test'] # the test cases
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point, # needed during evaluation
            "test": test, # needed during evaluation
        }
        return conversation

    def evaluate(self, conversation, completion):
        """ Given (conversation, completion), return boolean success of the completion. """
        imports = extract_imports(conversation['messages'][0]['content'])
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation['test']
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        success = result.success
        return success
