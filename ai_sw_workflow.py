"""
The software development workflow comprises eleven steps, as illustrated in the accompanying diagram. Each section of this paper provides a detailed explanation of a specific step:

    1. **Pseudocode Template**  
    2. **Recipe - High-Level Requirements**  
    3. **LLM API Interface**  
    4. **Pseudocode Requirements Translation**  
    5. **PSEUDO.code Requirements** - For target applications involving the *LLM challenges* above, custom intervention will be required at step 5 to manually edit the *Low-Level Requirements*.
    6. **Coding Policy**  
    7. **Coding Translation**  
    8. **Code Listing**  
    9. **Testing Policy**  
    10. **Code to Unit Test Translation**  

┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌────────┐
│ 2.Recipe    ┼─►#4──►│5.PSEUDO.code┼─►#7──►│   8.Code    ┼─►#10─►│11.uTest│
└─────────────┘   ▲   └─────────────┘   ▲   └─────────────┘   ▲   └────────┘
            ╔═ #1 ╚═══════╗       ┌─────┼───────┐       ┌─────┼───────┐          
            ║PSEUDO C TMPL║       │ 6.CodePolicy│       │ 9.TestPolicy│          
            ╚═════════════╝       └─────────────┘       └─────────────┘               
"""
import openai
import yaml
import sys
import logging
import subprocess
import os
import threading
import time
import re
from typing import List, Tuple, Optional, Iterable
from datetime import datetime
from argument_parser import ArgumentParser, XformType
from pathlib import Path
from enum import Enum
from typing import List, Dict
from datetime import datetime


ENCODING = 'utf-8'
FLOWDIAG_PSEUDO = """
┌─────────────┐ ╔═══╗ ┌─────────────┐       ┌─────────────┐       ┌────────┐
│ 2.Recipe    ┼──►#4─►│5.PSEUDO.code┼─►#7──►│   8.Code    ┼─►#10─►│11.uTest│
└─────────────┘ ╚═▲═╝ └─────────────┘   ▲   └─────────────┘   ▲   └────────┘
            ┌─#1 ─┼───────┐       ┌─────┼───────┐       ┌─────┼───────┐          
            │Pseudo Policy│       │ 6.CodePolicy│       │ 9.TestPolicy│          
            └─────────────┘       └─────────────┘       └─────────────┘          
"""
FLOWDIAG_CODE = """
┌─────────────┐       ┌─────────────┐ ╔═══╗ ┌─────────────┐       ┌────────┐
│ 2.Recipe    ┼─►#4──►│5.PSEUDO.code┼──►#7─►│   8.Code    ┼─►#10─►│11.uTest│
└─────────────┘   ▲   └─────────────┘ ╚═▲═╝ └─────────────┘   ▲   └────────┘
            ┌─#1 ─┼───────┐       ┌─────┼───────┐       ┌─────┼───────┐          
            │Pseudo Policy│       │ 6.CodePolicy│       │ 9.TestPolicy│          
            └─────────────┘       └─────────────┘       └─────────────┘          
"""
FLOWDIAG_TEST = """
┌─────────────┐       ┌─────────────┐       ┌─────────────┐ ╔═══╗ ┌─────────┐
│ 2.Recipe    ┼─►#4──►│5.PSEUDO.code┼─►#7──►│   8.Code    ┼─►#10─►│11.uTests│
└─────────────┘   ▲   └─────────────┘   ▲   └─────────────┘ ╚═▲═╝ └─────────┘
            ┌─#1 ─┼───────┐       ┌─────┼───────┐       ┌─────┼───────┐          
            │Pseudo Policy│       │ 6.CodePolicy│       │ 9.TestPolicy│          
            └─────────────┘       └─────────────┘       └─────────────┘          
"""

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

class PromptManager:
    PROMPT_PREFIX = ".prompt_"
    xform_mappings = [
        (XformType.TEST, "Translation CODE -> TEST", FLOWDIAG_TEST),
        (XformType.PSEUDO, "Translation HIGH_LVL_REQ -> PSEUDO", FLOWDIAG_PSEUDO),
        (None, "Translation PSEUDO -> CODE", FLOWDIAG_CODE)
    ]
    def __init__(self):
        self.clear()

    def clear(self) -> None:
        """
        Clears the list of messages.
        """
        self.system: Dict[str, str] = {}
        self.user_list: List[Dict[str, str]] = []
        self.variable_list = []

    def add_variable(self, variable_name: str, variable_value: str) -> None:
        """
        Creates the first message in the list with the given content.
        Args:
            system_role (str): The GPT role .
        """
        if variable_value and len(variable_value) > 0:
            var = (variable_name, variable_value)
            self.variable_list.append(var)
                       
    def system_prompt(self, system_role: str) -> None:
        """
        Creates the first message in the list with the given content.
        Args:
            system_role (str): The GPT role .
        """
        self.system = {"role": "system", "content": system_role}

    def append_prompt(self, content: str) -> None:
        """
        Appends a user message to the list with the given content.
        Args:
            content (str): The content for the user message.
        """
        self.user_list.append({"role": "user", "content": content})

    def get_prompt(self) -> List[Dict[str, str]]:
        """
        Retrieves the current list of messages.
        Returns:
            List[Dict[str, str]]: The current list of messages.
        """
        if self.system == {}:
            raise ValueError("System role is not set in class PromptManager().") 
        prompt: List[Dict[str, str]] = [self.system]
        if len(self.variable_list) > 0:
            content = "Perform substitution when the following variable names apear square brackets.\n"
            for variable_name, variable_value in self.variable_list:
                content += f"  - Let {variable_name} = \"{variable_value}\".\n"
            prompt.append({"role": "user", "content": content})
        prompt += self.user_list
        return prompt
    
    def prompt_fname(self, target: str) -> str:
        # CREATE PROMPT FILE NAME (.md)
        prompt_fname = Path(target).with_suffix(".md")
        prompt_fname = str(Path(prompt_fname).parent / f"{self.PROMPT_PREFIX}{Path(prompt_fname).name}")
        return prompt_fname

    def prompt_delete(self, target):
        prompt_fname = self.prompt_fname( target)
        if os.path.exists(target) :
            os.remove(target)
        if os.path.exists(prompt_fname):
            os.remove(prompt_fname)

    def prompt_compare_with_save(self, xform_type: XformType, target: str) -> bool:
        """
        Compares the content of a file to a given string, ignoring all whitespace differences.
        If `dest_fname` file does not exist - then False is returned

        :param file_path: Path to the file to compare.
        :param input_string: The string to compare with the file content.
        :return: True if the file content matches the string ignoring whitespace; False otherwise.
        """
        try:
            # CREATE HEADER/TITLE
            for xform, title_text, diagram_value in self.xform_mappings:
                if xform_type is xform or xform is None:
                    title = title_text
                    diagram = diagram_value
                    break
            prompt_text  = [f"# {title}\n\n"] + ["````\n"+ diagram + "````\n"]

            # APPEND PROMPT ITEMS - w/ markdown formatting
            prompt: List[Dict[str, str]] = self.get_prompt()
            for item in prompt:
                role = item["role"].capitalize()
                content = "\n".join(line for line in item["content"].strip().splitlines() if line.strip())
                prompt_text.append(f"### {role}\n\n{content}\n")
            prompt_text = "\n".join(prompt_text)

            # COMPARE PROMPTS
            prompt_fname = self.prompt_fname( target)
            is_exist = os.path.exists(target) # True if file exists
            is_match = False
            if is_exist and os.path.exists(prompt_fname):
                with open(prompt_fname, 'r', encoding=ENCODING) as file:
                    file_content = file.read()
                normalized_file_content = ' '.join(file_content.split())
                normalized_input_string = ' '.join(prompt_text.split())
                is_match = normalized_file_content == normalized_input_string

            # SAVE PROMPT FILE - for future comparison - process documentation
            if not is_match:
                with open(prompt_fname, 'w', encoding=ENCODING) as out:
                    out.write(prompt_text)
            return is_match        

        except Exception as e:
            print(f"An error occurred while comparing and saving the prompt: {e}")
            self.prompt_delete(target)
            raise


class LlmManager:
    """
    A class to manage a list of prompts for the OpenAI API.
    """
    MODEL_KEY_ENV_VARIABLE = "OPENAI_API_KEY"
    def __init__(self, temperature=0.1, max_tokens=4000, model='gpt-4o', target = "unnown"):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.target = target
        api_key = os.getenv(self.MODEL_KEY_ENV_VARIABLE)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{self.MODEL_KEY_ENV_VARIABLE}' is not set or empty. "
                "Please set this variable with your OpenAI API key."
            )
        self.client = openai.OpenAI(api_key=api_key)

    def _show_progress(self) -> None:
        """Print a period every second to indicate progress, with a counter."""
        self.running = True
        self.seconds = 0
        while self.running:
            self.seconds += 1
            n_tabs = 5 - (self.seconds+4) // 8
            tab = '\t'
            print(f"\t\r{self.seconds:>3}{'.' * self.seconds} {tab * n_tabs}{self.target}", 
                  end="", flush=True)
            time.sleep(1)

    def process_chat(self, prompt) -> Tuple[Optional[str], Tuple[int, int, str]]:
        """Process a chat and return the generated content and token usage."""
        try:
            # Start progress indicator
            self._progress_thread = threading.Thread(target=self._show_progress, daemon=True)
            self._progress_thread.start()            
            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    model=self.model
                )
            finally:
                self.running = False
                if self._progress_thread:
                    self._progress_thread.join()
            print(f"\t\r{self.seconds:>3}", end="", flush=True)
            usage = getattr(response, 'usage', None)
            if usage:
                return response.choices[0].message.content, (usage.prompt_tokens, usage.completion_tokens, self.model)
            return response.choices[0].message.content, (0, 0, self.model)
        except Exception as e:
            self.running = False
            print(f"\nprocess_chat() An error occurred: {e}")
            return None, (0, 0, self.model)


class LlmClient:
    LITERAL = "LITERAL"
    CODE_REF = "code_references"

    def __init__(self, policy_fname: str, code_fname: str = ""):
        try:
            # EXTRACT POLICY - a llm prompt must start with high level system role and user role
            with open(policy_fname, "r", encoding=ENCODING) as file:
                policy = yaml.safe_load(file)
            self.PROMPT_ELEMENTS    = policy["prompt_elements"]
            self.POSTFIX_ELEMENTS   = policy.get("postfix_elements", ["", ""])
            self.ROLE_SYSTEM    = policy["role_system"]
            self.ROLE_USER      = policy["role_user"]
            self.COMMENT_PREFIX = policy["comment_prefix"]
            self.COMMENT_POSTFIX= policy["comment_postfix"]
            self.TARGET_SUFFIXS = policy.get("testing_target_suffixs", [""])
            if self.TARGET_SUFFIXS == [""]:
                extension = os.path.splitext(code_fname)[1][1:]  # Removes the leading period
                self.TARGET_SUFFIXS = ["cpp", "hpp"]
            if self.POSTFIX_ELEMENTS == ["", ""]:
                self.POSTFIX_ELEMENTS = None                
        except Exception as e:  
            logging.error(
                "An error occurred while processing files:\n"
                "  Input files: %s\n"
                "  Error type: %s\n"
                "  Error details: %s",
                policy_fname, type(e).__name__, e
            )
            raise

    def strip_comments_python(self, source_code: str) -> str:
        """
        Strips all comments from the given source code.
        Args:
            source_code (str): The Python source code as a string.
        Returns:
            str: The source code with all comments removed.
        """
        no_single_line_comments = re.sub(r"#.*", "", source_code)
        no_comments = re.sub(r"(\"\"\".*?\"\"\"|'''.*?''')", "", no_single_line_comments, flags=re.DOTALL)
        stripped_code = "\n".join([line.rstrip() for line in no_comments.splitlines() if line.strip()])
        return stripped_code

    def comment_block(self, comment: str) -> str:
        return f"{self.COMMENT_PREFIX} {comment} {self.COMMENT_POSTFIX}"

    def add_file_references(self, prompt: PromptManager, references_str: str) -> None:
        """
        Processes file references from the recipe and appends them to the prompt.

        Args:
            recipe (dict): A dictionary containing the recipe with possible file references.
        """
        references = [
            line.lstrip("- ").strip()
            for line in references_str.splitlines()
            if line.strip()
        ]

        for ref_fname in references:
            target_base = Path(ref_fname)
            for suffix in self.TARGET_SUFFIXS:
                target_path = target_base.with_suffix(f".{suffix}")
                directory_path = os.path.dirname(target_path)
                file_name = os.path.basename(target_path)
                cmd = (
                    f"Use the following code as instructions to understand how to use the code: {target_path}.\n"
                    f"The code file named {file_name} from the subdirectory {directory_path}:\n"
                )
                try:
                    with open(target_path, 'r', encoding=ENCODING) as file:
                        reference = file.read()
                    reference = self.strip_comments_python(reference)    
                    prompt.append_prompt(
                        cmd + "\n```\n" + reference + "\n```\n"
                    )
                except FileNotFoundError:
                    print(f"File not found: {target_path}")
                    raise
                except Exception as e:
                    print(f"Error reading file {target_path}: {e}")
                    raise

    def create_prompt(self, prompt: PromptManager, xform_type: XformType, 
                       recipe_fname: str, code_fname: str) -> None:
        """
        Generates a code prompt by processing input YAML files and saves the results to specified files.
        Args:
            prompt (PromptManager): The prompt manager to handle prompt creation.
            xform_type (XformType): The type of transformation to apply.
            recipe_fname (str): The filename of the source YAML file.
            code_fname (str): The filename of the code file to be included in the prompt.
        Raises:
            Exception: If an error occurs during file processing.
        """
        try:
            prompt_elements = self.PROMPT_ELEMENTS
            prompt.system_prompt(self.ROLE_SYSTEM)
            prompt.append_prompt(self.ROLE_USER)
        
            # EXTRACT REQUIREMENTS - from req YAML using `prompt_elements` list 
            with open(recipe_fname, "r", encoding=ENCODING) as file:
                recipe = yaml.safe_load(file)               
                prompt.add_variable("TARGET_NAME", recipe.get("target_name", ""))
                for key, prefix in prompt_elements:

                    # ADD LITERAL - Insert element directly from recipe
                    if key == self.LITERAL:
                        prompt.append_prompt(prefix)

                    # ADD FILE REFERENCES - the whole file
                    elif key == self.CODE_REF and key in recipe:
                        self.add_file_references(prompt, recipe[key])

                    # ADD RECIPE ELEMENT - direct copy over from recipe
                    elif key in recipe:
                        if isinstance(recipe[key], Iterable) and not isinstance(recipe[key], (str, bytes)):
                            content = "\n".join(recipe[key] )  
                        else:
                            content = recipe[key]
                        prompt.append_prompt(prefix + "\n" + content)

            # INCLUDE CODE - to be tested
            if xform_type is XformType.TEST:
                target_base = Path(code_fname)
                for suffix in self.TARGET_SUFFIXS:
                    suffix = "." + suffix
                    target_path = target_base.with_suffix(suffix)
                    with open(target_path, 'r', encoding=ENCODING) as file:
                        target = file.read()
                    prefix =  "You are to write Unit Tests for the following source code."
                    prefix += f"The source code comes from the file {target_path}"
                    prompt.append_prompt(prefix + "\n```\n" + target + "\n```\n")

        except Exception as e:  # Replace SpecificException with the specific type you're expecting.
            logging.error(
                "An error occurred while processing files:\n"
                "  Input files: %s\n"
                "  Output file: %s\n"
                "  Error type: %s\n"
                "  Error details: %s",
                recipe_fname, code_fname, type(e).__name__, e
            )
            raise

    def process_response_to_yaml(self, response: str, recipe_fname: str, target: str) -> str:
        """
        Post-process the response
        Args:
            response (str): The response to be processed. It should be a non-empty string.
            xform_type (XformType): The type of transformation to be applied. This determines how the response will be processed.
            recipe_fname (str): The name of the source file from which the response was generated.
        Raises:
            Exception: If an error occurs during file processing or if the response is empty.
        Returns:
            str: The processed response after applying the specified transformation.
        """
        try:
            # EXTRACT YAML from response
            response = re.sub(r'^.*?```yaml\n', '', response, flags=re.DOTALL)
            #find the last occurrence of the triple single quotes (''') in a string and remove everything after it,
            # you can use a regular expression with a negative lookahead to ensure it matches the last occurrence.
            response = re.sub(r"```(?!.*```).*$", "", response, flags=re.DOTALL)
            # ADD PROMPT ELEMENTS
            literals = "\n"
            prompt_elements = self.POSTFIX_ELEMENTS
            with open(recipe_fname, "r", encoding=ENCODING) as file:
                recipe = yaml.safe_load(file)
                if prompt_elements is not None:
                    for key, _ in prompt_elements:
                        if key in recipe:
                            if isinstance(recipe[key], Iterable) and not isinstance(recipe[key], (str, bytes)):
                                content = "\n".join(recipe[key])
                            else:
                                content = recipe[key]
                            if isinstance(content, bool):
                                literals += f"\n  {key}: {content}"
                            else:
                                literals += f"\n  {key}: |\n"
                                literals += "\n".join([f"    {line}" for line in content.splitlines()])
                                literals += "\n"
            return response + literals

        except Exception as e:
            error_type = type(e).__name__
            print(f"An error occurred while processing files:\n  Input files:  {recipe_fname}\n")
            print(f"  Error type: {error_type}\n  Error details: {e}")
            raise


def main_xform(policy_fname: str, recipe_fname: str, code_fname: str, dest_fname: str, 
               xform_type: XformType, temperature: float, max_tokens: int, model: str) -> None:
    prompt = PromptManager()
    try:
        # CREATE LLM PROMPT
        client = LlmClient(policy_fname=policy_fname, code_fname=code_fname)
        filename_stem = Path(dest_fname).stem
        prompt.add_variable("DATE", datetime.now().strftime("%Y-%m-%d"))
        prompt.add_variable("RECIPE_FNAME", recipe_fname)
        prompt.add_variable("TARGET_FNAME", dest_fname)
        prompt.add_variable("CODE_FNAME", code_fname)
        prompt.add_variable("FILENAME_STEM", filename_stem)
        client.create_prompt(prompt, xform_type=xform_type, 
                             recipe_fname=recipe_fname, code_fname=code_fname)
        if prompt.prompt_compare_with_save(xform_type=xform_type, target=dest_fname):
            if os.path.exists(dest_fname):
                print(f"\t\t {dest_fname} - Skipping transform, prompts match & target exists.")
                return  # KLUDGE - mid function abort
        
        # PROCESS LLM PROMPT
        llm_mgr = LlmManager(temperature=temperature, max_tokens=max_tokens, model=model, target = dest_fname)
        response, tokens = llm_mgr.process_chat(prompt.get_prompt())
        if response is not None:
            # POST PROCESS RESPONSE
            response_yaml = client.process_response_to_yaml(response, recipe_fname=recipe_fname, target = dest_fname)
            token_usage = f" TOKENS: {tokens[0] + tokens[1]} (of:{max_tokens}) = {tokens[0]} + {tokens[1]}(prompt+return)"
            header = client.comment_block( f"{token_usage} -- MODEL: {tokens[2]}") + "\n"
            header += client.comment_block(f"policy: {policy_fname}") + "\n"
            header += client.comment_block(f"code: {code_fname}") + "\n"
            header += client.comment_block(f"dest: {dest_fname}") + "\n"
            print(f"{token_usage}  \t {dest_fname}")

            # SAVE EACH KEY - to unique file
            target_base = Path(dest_fname)
            content_yaml = yaml.safe_load(response_yaml)
            for key, content in content_yaml.items():
                target_path = target_base.with_suffix("." + key)
                with open(target_path, 'w', encoding=ENCODING) as file:
                    file.write(header + content)
        else:
            prompt.prompt_delete(dest_fname)

    except Exception as e:
        e_msg = f"An error occurred while processing files:\n  Input files: {policy_fname}\n  Output file: {code_fname}\n  Error details: {e}"
        print(f"ERROR THROWN {e_msg}")
        prompt.prompt_delete(dest_fname)
        raise


def main():
    # Configure logging to show DEBUG level and above
    logging.basicConfig(level=logging.ERROR)    
    install_and_import("openai")
    install_and_import("yaml")
    parser = ArgumentParser()
    args = parser.parse()

    # IS XFORM REQUIRED
    # args.xform = ".cpp" in args.code and XformType.CODE or XformType.PSEUDO
    is_xform_enabled = True
    TEST_ENABLE = "test_enable"
    if args.xform is XformType.TEST: 
        with open(args.recipe, "r", encoding=ENCODING) as file:
            rules = yaml.safe_load(file)
            is_xform_enabled = "true" in str(rules.get(TEST_ENABLE, "true")).lower()
            if not is_xform_enabled:
                # print(f"Skipping TEST generation for {args.recipe} as {TEST_ENABLE} is FALSE.")
                return # KLUDGE - mid function abort

    if args is not None and is_xform_enabled:
        main_xform(policy_fname=args.policy, recipe_fname=args.recipe, code_fname=args.code, dest_fname=args.dest, 
                xform_type=args.xform, temperature=args.temperature, max_tokens=args.maxtokens, model=args.model)
    else:
        print("Error: Failed to parse arguments.")

if __name__ == "__main__":
    main()

###########################################################################
# TODO - Policy cleanup - less verbose - more specific
# TODO - fix fibinacci test - infinite loop  [ RUN      ] FibonacciTest.ThrowsInvalidArgumentForNegativeIndex

### MAKEFILE
# TODO - Trigger builld if RECIPE changes - %$(TEST_SUFFIX): %$(CODE_SUFFIX) %$(CODE_SUFFIX)
# TODO - if GPT prompt fails (i.e. timeout) - make is broken until you manually fix it delete stale target file..  
#          just delete prompt?             
