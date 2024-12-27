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
TEMPLFLOWDIAG = """
┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌────────┐
│ 2.HiLvlRqmt ┼─►#4──►│ 5.LoLvlRqmt ┼─►#7──►│   8.Code    ┼─►#10─►│11.uTest│
└─────────────┘   ▲   └─────────────┘   ▲   └─────────────┘   ▲   └────────┘
            ╔═ #1 ╚═══════╗       ┌─────┼───────┐       ┌─────┼───────┐          
            ║PSEUDO C TMPL║       │ 6.CodePolicy│       │ 9.TestPolicy│          
            ╚═════════════╝       └─────────────┘       └─────────────┘
"""

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def strip_comments_python(source_code: str) -> str:
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


class PromptManager:
    PROMPT_PREFIX = ".prompt_"
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
    
    def prompt_compare_with_save(self, target: str) -> bool:
        """
        Compares the content of a file to a given string, ignoring all whitespace differences.
        If `dest_fname` file does not exist - then False is returned

        :param file_path: Path to the file to compare.
        :param input_string: The string to compare with the file content.
        :return: True if the file content matches the string ignoring whitespace; False otherwise.
        """
        try:
            prompt: List[Dict[str, str]] = self.get_prompt()
            prompt_fname = Path(target).with_suffix(".md")
            prompt_fname = str(Path(prompt_fname).parent / f"{self.PROMPT_PREFIX}{Path(prompt_fname).name}")
            prompt_text = ["````\n"+ TEMPLFLOWDIAG + "````\n"]
            for item in prompt:
                role = item["role"].capitalize()
                content = "\n".join(line for line in item["content"].strip().splitlines() if line.strip())
                prompt_text.append(f"### {role}\n\n{content}\n")
            prompt_text = "\n".join(prompt_text)

            is_match = False
            is_exist = os.path.exists(target) # True if file exists
            if is_exist:
                if os.path.exists(prompt_fname):
                    with open(prompt_fname, 'r', encoding=ENCODING) as file:
                        file_content = file.read()
                    normalized_file_content = ' '.join(file_content.split())
                    normalized_input_string = ' '.join(prompt_text.split())
                    is_match = normalized_file_content == normalized_input_string
            if not is_match:
                with open(prompt_fname, 'w', encoding=ENCODING) as out:
                    out.write(prompt_text)
            return is_match        

        except Exception as e:
            print(f"An error occurred: {e}")
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
            # print("")  # Clear progress indicator
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
    POLICY_PAIRS = "key_prefix_pairs"

    def __init__(self):
        pass

    def _extract_code_from_response(self, response: str) -> str:
        """Extract and clean code from a response string."""
        response = re.sub(r'^.*?\`\`\`', '', response, flags=re.DOTALL)
        response = re.sub(r'```.*', '', response, flags=re.DOTALL)
        response = re.sub(r'^([\`]{3})', '#', response, flags=re.MULTILINE)
        response = re.sub(r'^python', '#', response, flags=re.MULTILINE)
        return response

    def create_prompt(self, prompt: PromptManager, xform_type: XformType, policy_fname: str, source_fname: str, 
                dest_fname: str, code_fname: str) -> None:
        """
        Process the input YAML files to generate a code prompt and save results to specified files.
        Parameters:
        """
        try:
            # EXTRACT POLICY - a llm prompt must start with high level system role and user role
            with open(policy_fname, "r", encoding=ENCODING) as file:
                policy = yaml.safe_load(file)
                key_prefix_pairs = policy[self.POLICY_PAIRS]
                prompt.system_prompt(policy["role_system"])
                prompt.append_prompt(policy["role_user"])
            prompt.add_variable("SOURCE_FNAME", source_fname)
            prompt.add_variable("TARGET_FNAME", dest_fname)
            prompt.add_variable("CODE_FNAME", code_fname)
        
            # EXTRACT REQUIREMENTS - from req YAML using `key_prefix_pairs` list 
            with open(source_fname, "r", encoding=ENCODING) as file:
                rules = yaml.safe_load(file)               
                prompt.add_variable("BASE_FNAME", rules.get("base_fname", ""))
                prompt.add_variable("TARGET_NAME", rules.get("target_name", ""))
                for key, prefix in key_prefix_pairs:
                    if key == self.LITTERAL:
                        prompt.append_prompt(prefix)
                    elif key == self.CODE_REF and key in rules : #and xform_type is XformType.TEST:
                        references_str = rules[key]
                        references = [line.lstrip("- ").strip() for line in references_str.splitlines() if line.strip()]
                        for ref_fname in references:
                            module = ref_fname.split('/')[0]
                            fname  = ref_fname.split('/')[1]
                            cmd = (f"Use the following as instructions to understand how to use the package: {module}.\n"
                                    f"The package's module is stored in a file named {fname} from the subdirectory {module} :\n")
                            with open(ref_fname, 'r', encoding=ENCODING) as file:
                                reference = file.read()
                            if code_fname.endswith(".py"):
                                reference = strip_comments_python(reference)
                            prompt.append_prompt(cmd + reference)
                    elif key in rules and not key.startswith("_"):
                            prompt.append_prompt(prefix + "\n" + rules[key])

            # INCLUDE CODE - to be tested
            if xform_type is XformType.TEST: 
                with open(code_fname, "r", encoding=ENCODING) as file:
                    code = file.read()
                    if code_fname.endswith(".py"):
                        code = strip_comments_python(code)
                prefix = f"Write Unit Test for the following code which comes from a file named: {code_fname}:\n"    
                prompt.append_prompt(prefix + "\n\n\n```" + code + "\n\n\n```")

        except Exception as e:
            error_type = type(e).__name__
            print(f"An error occurred while processing files:\n  Input files: {policy_fname}, {source_fname}\n")
            print(f"  Output file: {dest_fname}\n  Error type: {error_type}\n  Error details: {e}")
            raise

    def post_process(self, response: str, xform_type: XformType, policy_fname: str, source_fname: str, 
                dest_fname: str, code_fname: str) -> str:
        try:
            result = ""
            if not xform_type is XformType.PSEUDO:
                result += self._extract_code_from_response(response) 
            else:
                # XformType.PSEUDO - remove code blocks from response
                response = response.replace("```yaml", "   ")
                response = response.replace("```", "   ")
                result +=  '\n' +  response

                # LITERAL reqirements (which are prefix w/ underscore) 
                # copy them directly into PSEUDO yaml file
                with open(policy_fname, "r", encoding=ENCODING) as file:
                    policy = yaml.safe_load(file)
                    key_prefix_pairs = policy[self.POLICY_PAIRS]
                with open(source_fname, "r", encoding=ENCODING) as file:
                    rules = yaml.safe_load(file)
                    for _key, prefix in key_prefix_pairs:
                        key = _key[1:]
                        if _key.startswith("_") and key in rules:
                            content = rules[key]
                            if type(content) is bool:
                                result += f"\n{key}: {content}"
                            else:
                                result += f"\n{key}: |\n"
                                # result += "  - " + prefix + "\n  "
                                result += "\n".join([f"  {line}" for line in content.splitlines()])
                                result += "\n"
            if len(result) == 0:
                raise RuntimeError("Failed to generate response.")
            return result

        except Exception as e:
            error_type = type(e).__name__
            print(f"An error occurred while processing files:\n  Input files: {policy_fname}, {source_fname}\n")
            print(f"  Output file: {dest_fname}\n  Error type: {error_type}\n  Error details: {e}")
            raise

def main():
    install_and_import("openai")
    install_and_import("yaml")
    # TODO - how to deal with BASE NAME and TARGET NAME
    # TODO - makefile to process subdirectories first before base directory
    try:
        # ARGUMENT PARSING
        parser = ArgumentParser()
        args = parser.parse()
        if args is None:
            print("Error: Failed to parse arguments.")
            return # KLUDGE - mid function abort
        
        # IS XFORM REQUIRED
        is_xform_enabled = True
        TEST_ENABLE="test_enable"
        if args.xform is XformType.TEST: 
            with open(args.source, "r", encoding=ENCODING) as file:
                rules = yaml.safe_load(file)
                is_xform_enabled = "true" in str(rules.get(TEST_ENABLE, "true")).lower()
                if not is_xform_enabled:
                    print(f"Skipping TEST generation for {args.source} as {TEST_ENABLE} is not set to True.")
                    return # KLUDGE - mid function abort

        prompt = PromptManager()
        client = LlmClient()
        client.create_prompt( prompt, xform_type=args.xform, policy_fname=args.policy, 
            source_fname=args.source, dest_fname=args.dest, code_fname=args.code, )
        if prompt.prompt_compare_with_save(args.dest):
            print(f"\t\t {args.dest} - Skipping generation, prompts match.")
            return  # KLUDGE - mid function abort
        
        llm_mgr = LlmManager(temperature=args.temperature, max_tokens=args.maxtokens, model=args.model)
        response, tokens = llm_mgr.process_chat(prompt.get_prompt())
        response = client.post_process( response, xform_type=args.xform, policy_fname=args.policy, 
            source_fname=args.source, dest_fname=args.dest, code_fname=args.code, )
        token_usage = f"# TOKENS: {tokens[0] + tokens[1]} (of:{args.maxtokens}) = {tokens[0]} + {tokens[1]}(prompt+return) -- MODEL: {tokens[2]}"
        token_usage2 = f"TOKENS: {tokens[0] + tokens[1]} (of:{args.maxtokens}) = {tokens[0]} + {tokens[1]}(prompt+return)"
        print(f"{args.dest} - {token_usage2}")
        result = f"{token_usage}\n{response}"
        with open(args.dest, 'w', encoding=ENCODING) as out:
            out.write(result)

    except Exception as e:
        e_msg = f"An error occurred while processing files:\n  Input files: {args.policy}\n  Output file: {args.code}\n  Error details: {e}"
        print(f"ERROR THROWN {e_msg}")
        raise

if __name__ == "__main__":
    main()
