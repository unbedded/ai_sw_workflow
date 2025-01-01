"""
The software development workflow comprises eleven steps, as illustrated in the accompanying diagram. Each section of this paper provides a detailed explanation of a specific step:

    1. **Pseudocode Template**  
    2. **High-Level Requirements**  
    3. **LLM API Interface**  
    4. **High-Level to Low-Level Requirements Translation**  
    5. **Low-Level Requirements** - For target applications involving the *LLM challenges* above, custom intervention will be required at step 5 to manually edit the *Low-Level Requirements*.
    6. **Coding Policy**  
    7. **Low-Level Requirements to Code Translation**  
    8. **Code Listing**  
    9. **Testing Policy**  
    10. **Code to Unit Test Translation**  

┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌────────┐
│ 2.HiLvlRqmt ┼─►#4──►│ 5.LoLvlRqmt ┼─►#7──►│   8.Code    ┼─►#10─►│11.uTest│
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
from typing import List, Tuple, Optional
from datetime import datetime
from argument_parser import ArgumentParser, XformType
from pathlib import Path
from enum import Enum
from typing import List, Dict

ENCODING = 'utf-8'
FLOWDIAG_PSEUDO = """
┌─────────────┐ ╔═══╗ ┌─────────────┐       ┌─────────────┐       ┌────────┐
│ 2.HiLvlRqmt ┼──►#4─►│ 5.LoLvlRqmt ┼─►#7──►│   8.Code    ┼─►#10─►│11.uTest│
└─────────────┘ ╚═▲═╝ └─────────────┘   ▲   └─────────────┘   ▲   └────────┘
            ┌─#1 ─┼───────┐       ┌─────┼───────┐       ┌─────┼───────┐          
            │Pseudo Policy│       │ 6.CodePolicy│       │ 9.TestPolicy│          
            └─────────────┘       └─────────────┘       └─────────────┘          
"""
FLOWDIAG_CODE = """
┌─────────────┐       ┌─────────────┐ ╔═══╗ ┌─────────────┐       ┌────────┐
│ 2.HiLvlRqmt ┼─►#4──►│ 5.LoLvlRqmt ┼──►#7─►│   8.Code    ┼─►#10─►│11.uTest│
└─────────────┘   ▲   └─────────────┘ ╚═▲═╝ └─────────────┘   ▲   └────────┘
            ┌─#1 ─┼───────┐       ┌─────┼───────┐       ┌─────┼───────┐          
            │Pseudo Policy│       │ 6.CodePolicy│       │ 9.TestPolicy│          
            └─────────────┘       └─────────────┘       └─────────────┘          
"""
FLOWDIAG_TEST = """
┌─────────────┐       ┌─────────────┐       ┌─────────────┐ ╔═══╗ ┌─────────┐
│ 2.HiLvlRqmt ┼─►#4──►│ 5.LoLvlRqmt ┼─►#7──►│   8.Code    ┼─►#10─►│11.uTests│
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
            content = f"Perform substitution when the following variable names apear square brackets.\n"
            for variable_name, variable_value in self.variable_list:
                content += f"  - {variable_name} = \"{variable_value}\".\n"
            prompt.append({"role": "user", "content": content})
        prompt += self.user_list
        return prompt
    
    def prompt_compare_with_save(self, xform_type: XformType, target: str) -> bool:
        """
        Compares the content of a file to a given string, ignoring all whitespace differences.
        If `dest_fname` file does not exist - then False is returned

        :param file_path: Path to the file to compare.
        :param input_string: The string to compare with the file content.
        :return: True if the file content matches the string ignoring whitespace; False otherwise.
        """
        try:

            # CREATE PROMPT FILE NAME (.md)
            prompt: List[Dict[str, str]] = self.get_prompt()
            prompt_fname = Path(target).with_suffix(".md")
            prompt_fname = str(Path(prompt_fname).parent / f"{self.PROMPT_PREFIX}{Path(prompt_fname).name}")

            # CREATE HEADER/TITLE
            for xform, title_text, diagram_value in self.xform_mappings:
                if xform_type is xform or xform is None:
                    title = title_text
                    diagram = diagram_value
                    break
            prompt_text  = [f"# {title}\n\n"] + ["````\n"+ diagram + "````\n"]

            # APPEND PROMPT ITEMS - w/ markdown formatting
            for item in prompt:
                role = item["role"].capitalize()
                content = "\n".join(line for line in item["content"].strip().splitlines() if line.strip())
                prompt_text.append(f"### {role}\n\n{content}\n")
            prompt_text = "\n".join(prompt_text)

            # COMPARE PROMPTS
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
            return False

class LlmManager:
    """
    A class to manage a list of prompts for the OpenAI API.
    """
    MODEL_KEY_ENV_VARIABLE = "OPENAI_API_KEY"
    def __init__(self, temperature=0.1, max_tokens=4000, model='gpt-4o'):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
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
        seconds = 0
        while self.running:
            seconds += 1
            print(f"\r{seconds:>3} {'.' * seconds}", end="", flush=True)
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
            usage = getattr(response, 'usage', None)
            if usage:
                return response.choices[0].message.content, (usage.prompt_tokens, usage.completion_tokens, self.model)
            return response.choices[0].message.content, (0, 0, self.model)
        except Exception as e:
            self.running = False
            print(f"\nprocess_chat() An error occurred: {e}")
            return None, (0, 0, self.model)


class LlmClient:
    LITTERAL = "LITERAL"
    CODE_REF = "code_references"

    def __init__(self, policy_fname: str, code_fname: str = ""):
        try:
            # EXTRACT POLICY - a llm prompt must start with high level system role and user role
            with open(policy_fname, "r", encoding=ENCODING) as file:
                policy = yaml.safe_load(file)
            self.POLICY_PAIRS   = policy["key_prefix_pairs"]
            self.ROLE_SYSTEM    = policy["role_system"]
            self.ROLE_USER      = policy["role_user"]
            self.COMMENT_PREFIX = policy["comment_prefix"]
            self.COMMENT_POSTFIX= policy["comment_postfix"]
            self.TARGET_SUFFIXS = policy.get("testing_target_suffixs", [""])
            if self.TARGET_SUFFIXS == [""]:
                extension = os.path.splitext(code_fname)[1][1:]  # Removes the leading period
                self.TARGET_SUFFIXS = ["cpp", "hpp"]
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

    def create_prompt(self, prompt: PromptManager, xform_type: XformType, 
                       source_fname: str, code_fname: str) -> None:
        """
        Generates a code prompt by processing input YAML files and saves the results to specified files.
        Args:
            prompt (PromptManager): The prompt manager to handle prompt creation.
            xform_type (XformType): The type of transformation to apply.
            source_fname (str): The filename of the source YAML file.
            code_fname (str): The filename of the code file to be included in the prompt.
        Raises:
            Exception: If an error occurs during file processing.
        """
        try:
<<<<<<< HEAD
            key_prefix_pairs = self.POLICY_PAIRS
            prompt.system_prompt(self.ROLE_SYSTEM)
            prompt.append_prompt(self.ROLE_USER)
=======
            # EXTRACT POLICY - a llm prompt must start with high level system role and user role
            with open(policy_fname, "r", encoding=ENCODING) as file:
                policy = yaml.safe_load(file)
                prompt.system_prompt(policy["role_system"])
                prompt.append_prompt(policy["role_user"])
                key_prefix_pairs = policy[self.POLICY_PAIRS]

>>>>>>> 8803b0ce045145edce068ce23a165a9a91ffd552
        
            # EXTRACT REQUIREMENTS - from req YAML using `key_prefix_pairs` list 
            with open(source_fname, "r", encoding=ENCODING) as file:
                rules = yaml.safe_load(file)               
                for rule in rules:
                    if type(rule) is not str :
                        print(f"{rule} type {type(rule)}")
                        # msg = f"Invalid type for {self.POLICY_PAIRS} - key: {key} in {policy_fname}."
                        # print(msg)
                        # raise ValueError(msg)
                    # if rule not in [self.LITTERAL, self.CODE_REF, self.POLICY_PAIRS]:
                    #     msg = f"Invalid key in {source_fname}: {rule}."
                    #     print(msg)
                    #     raise ValueError(msg)
                # for key, prefix in key_prefix_pairs:

                prompt.add_variable("TARGET_NAME", rules.get("target_name", ""))
                for key, prefix in key_prefix_pairs:
                    if key == self.LITTERAL:
                        prompt.append_prompt(prefix)
                    elif key == self.CODE_REF and key in rules:
                        references_str = rules[key]
                        references = [line.lstrip("- ").strip() for line in references_str.splitlines() if line.strip()]
                        for ref_fname in references:
                            target_base = Path(ref_fname)
                            for suffix in self.TARGET_SUFFIXS:
                                suffix = "." + suffix
                                target_path = target_base.with_suffix(suffix)
                                module = str(target_path).split('/')[0]
                                fname  = str(target_path).split('/')[1]
                                cmd = (f"Use the following code as instructions to understand how to use the C++ code: {target_path}.\n"
                                        f"The code code file named {fname} from the subdirectory {module} :\n")
                                with open(target_path, 'r', encoding=ENCODING) as file:
                                    reference = file.read()
                                prompt.append_prompt(cmd + "\n```\n" + reference + "\n```\n")
                    elif key in rules and not key.startswith("_"):
                            prompt.append_prompt(prefix + "\n" + rules[key])

                            # if code_fname.endswith(".py"):
                            #     module = ref_fname.split('/')[0]
                            #     fname  = ref_fname.split('/')[1]
                            #     cmd = (f"Use the following as instructions to understand how to use the package: {module}.\n"
                            #             f"The package's module is stored in a file named {fname} from the subdirectory {module} :\n")
                            #     with open(ref_fname, 'r', encoding=ENCODING) as file:
                            #         reference = file.read()
                            #     reference = self.strip_comments_python(reference)
                            #     prompt.append_prompt(cmd +"\n```\n" + reference +"\n```\n" )
                            # elif code_fname.endswith(".cpp"):
                            #     target_base = Path(ref_fname)
                            #     for suffix in self.TARGET_SUFFIXS:
                            #         suffix = "." + suffix
                            #         target_path = target_base.with_suffix(suffix)
                            #         module = str(target_path).split('/')[0]
                            #         fname  = str(target_path).split('/')[1]
                            #         cmd = (f"Use the following code as instructions to understand how to use the C++ code: {target_path}.\n"
                            #                f"The code code file named {fname} from the subdirectory {module} :\n")
                            #         with open(target_path, 'r', encoding=ENCODING) as file:
                            #             reference = file.read()
                            #         prompt.append_prompt(cmd + "\n```\n" + reference + "\n```\n")
                            # else:
                            #     print(f"ERROR - Unknown file extension: {code_fname}")  

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
                source_fname, code_fname, type(e).__name__, e
            )
            raise
    def reponse_to_dict(self, response: str, source_fname: str) -> str:
        """
        Post-process the response
        Args:
            response (str): The response to be processed.
            xform_type (XformType): The type of transformation to be applied.
            source_fname (str): The name of the source file.
        Raises:
            Exception: If an error occurs during file processing - or if response is empty.
        """        
        try:
            # EXTRACT YAML from response
            response = re.sub(r'^.*?```yaml\n', '', response, flags=re.DOTALL)
            response = re.sub(r'```.*$',        '', response, flags=re.DOTALL)

            # ADD LITERAL values (which are prefixed with underscore)
            literals = "\n"
            key_prefix_pairs = self.POLICY_PAIRS
            with open(source_fname, "r", encoding=ENCODING) as file:
                rules = yaml.safe_load(file)
                for _key, _ in key_prefix_pairs:
                    key = _key[1:]
                    if _key.startswith("_") and key in rules:
                        # YIKES _ just yaml.dump() did horrible things to the yaml - so I had to do it manually
                        content = rules[key]
                        if isinstance(content, bool):
                            literals += f"\n{key}: {content}"
                        else:
                            literals += f"\n{key}: |\n"
                            literals += "\n".join([f"  {line}" for line in content.splitlines()])
                            literals += "\n"
            return response + literals

        except Exception as e:
            error_type = type(e).__name__
            print(f"An error occurred while processing files:\n  Input files:  {source_fname}\n")
            print(f"  Error type: {error_type}\n  Error details: {e}")
            raise


def main_xform(policy_fname: str, source_fname: str, code_fname: str, dest_fname: str, 
               xform_type: XformType, temperature: float, max_tokens: int, model: str) -> None:
    try:
        # CREATE LLM PROMPT
        prompt = PromptManager()
        client = LlmClient(policy_fname=policy_fname, code_fname=code_fname)
        prompt.add_variable("SOURCE_FNAME", source_fname)
        prompt.add_variable("TARGET_FNAME", dest_fname)
        prompt.add_variable("CODE_FNAME", code_fname)
        client.create_prompt(prompt, xform_type=xform_type, 
                             source_fname=source_fname, code_fname=code_fname)
        if prompt.prompt_compare_with_save(xform_type=xform_type, target=dest_fname):
            print(f"\t\t {dest_fname} - Skipping generation, prompts match.")
            return  # KLUDGE - mid function abort
        
        # PROCESS LLM PROMPT
        llm_mgr = LlmManager(temperature=temperature, max_tokens=max_tokens, model=model)
        response, tokens = llm_mgr.process_chat(prompt.get_prompt())

        # POST PROCESS RESPONSE
        response_processed = client.reponse_to_dict(response, source_fname=source_fname)
        token_usage = f"TOKENS: {tokens[0] + tokens[1]} (of:{max_tokens}) = {tokens[0]} + {tokens[1]}(prompt+return)"
        header = client.comment_block( f"{token_usage} -- MODEL: {tokens[2]}") + "\n"
        header += client.comment_block(f"policy: {policy_fname}") + "\n"
        header += client.comment_block(f"code: {code_fname}") + "\n"
        header += client.comment_block(f"dest: {dest_fname}") + "\n"
        print(f"{dest_fname} -> {token_usage}")

        if xform_type is XformType.PSEUDO:
            with open(dest_fname, 'w', encoding=ENCODING) as file:
                file.write(header + response_processed)
        else:
            target_base = Path(dest_fname)
            response_yaml = yaml.safe_load(response_processed)
            for key, target in response_yaml.items():
                target_path = target_base.with_suffix("." + key)
                with open(target_path, 'w', encoding=ENCODING) as file:
                    file.write(header + target)

    except Exception as e:
        e_msg = f"An error occurred while processing files:\n  Input files: {policy_fname}\n  Output file: {code_fname}\n  Error details: {e}"
        print(f"ERROR THROWN {e_msg}")
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
        with open(args.source, "r", encoding=ENCODING) as file:
            rules = yaml.safe_load(file)
            is_xform_enabled = "true" in str(rules.get(TEST_ENABLE, "true")).lower()
            if not is_xform_enabled:
                print(f"Skipping TEST generation for {args.source} as {TEST_ENABLE} is not set to True.")
                return # KLUDGE - mid function abort

    if args is not None and is_xform_enabled:
        main_xform(policy_fname=args.policy, source_fname=args.source, code_fname=args.code, dest_fname=args.dest, 
                xform_type=args.xform, temperature=args.temperature, max_tokens=args.maxtokens, model=args.model)
    else:
        print("Error: Failed to parse arguments.")

if __name__ == "__main__":
    main()

###########################################################################
# TODO - makefile to process subdirectories first before base directory
# TODO - pytest coverage  w/ summary report 
# TODO - pytest autorun
# TODO - Date is wrong Date: 2023-10-05
# TODO - Fix references for non-python files
# TODO - change _req.yaml ext to recipe.yaml
# TODO - make gtest as its own rule
# TODO - change source_fname to recipe_fname

# TODO - xform needs to know target language
# TODO - makefile gtest enable
# TODO - makefile not fail if no tests
# TODO - #1?
# TODO - #2?
