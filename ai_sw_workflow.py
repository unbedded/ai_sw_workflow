## TODO - if only test req was edited - then only remake test
## TODO - if xform.py is newer than _req.yaml  - warn user if
## TODO - why do these fail? 
             # ("/home/preact/Music/240110_13WPM.wav", 20, 10, 20, 13),
             # ("/home/preact/Music/220112_10WPM.wav", 20, 10, 20, 10),
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
from argument_parser.argument_parser import ArgumentParser, XformType
from pathlib import Path
from enum import Enum

# def install_and_import(package):
#     try:
#         __import__(package)
#     except ImportError:
#         print(f"Installing {package}...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package])

class LlmClient:
    WORKFLOW_DIR = "ai_sw_workflow"
    POLICY_DIR = "policy"
    ENCODING = 'utf-8'
    PROMPT_PREFIX = ".prompt_"
    LITTERAL = "LITERAL"


    def __init__(self, temperature=0.1, max_tokens=4000, model='gpt-4o'):
        MODEL_KEY_ENV_VARIABLE = "OPENAI_API_KEY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model

        api_key = os.getenv(MODEL_KEY_ENV_VARIABLE)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{MODEL_KEY_ENV_VARIABLE}' is not set or empty. "
                "Please set this variable with your OpenAI API key."
            )
        self.client = openai.OpenAI(api_key=api_key)

    def _show_progress(self) -> None:
        """Print a period every second to indicate progress, with a counter."""
        self.running = True
        seconds = 0
        # print("\nProgress:")
        while self.running:
            seconds += 1
            print(f"\r\t{seconds:>3} {'.' * seconds}", end="", flush=True)
            time.sleep(1)

    def process_chat(self, messages: List[dict]) -> Tuple[Optional[str], Tuple[int, int, str]]:
        """Process a chat and return the generated content and token usage."""
        try:
            # Start progress indicator
            self._progress_thread = threading.Thread(target=self._show_progress, daemon=True)
            self._progress_thread.start()            
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    model=self.model
                )
            finally:
                self.running = False
                if self._progress_thread:
                    self._progress_thread.join()
            print("")  # Clear progress indicator
            usage = getattr(response, 'usage', None)
            if usage:
                return response.choices[0].message.content, (usage.prompt_tokens, usage.completion_tokens, self.model)
            return response.choices[0].message.content, (0, 0, self.model)
        except Exception as e:
            self.running = False
            print(f"\nprocess_chat() An error occurred: {e}")
            return None, (0, 0, self.model)

    def extract_code_from_response(self, response: str) -> str:
        """Extract and clean code from a response string."""
        response = re.sub(r'^.*?\`\`\`', '', response, flags=re.DOTALL)
        response = re.sub(r'```.*', '', response, flags=re.DOTALL)
        response = re.sub(r'^([\`]{3})', '#', response, flags=re.MULTILINE)
        response = re.sub(r'^python', '#', response, flags=re.MULTILINE)
        return response

    def compare_file_to_string(self, file_path: str, prompt: List[dict]) -> bool:
            """
            Compares the content of a file to a given string, ignoring all whitespace differences.

            :param file_path: Path to the file to compare.
            :param input_string: The string to compare with the file content.
            :return: True if the file content matches the string ignoring whitespace; False otherwise.
            """
            if not os.path.exists(file_path):
                # print(f"File does not exist: {file_path}")
                return False

            try:
                input_string = ", ".join(map(str, prompt))
                input_string = input_string.replace("\\n", "\n")

                with open(file_path, 'r', encoding=self.ENCODING) as file:
                    file_content = file.read()
                
                normalized_file_content = ''.join(file_content.split())
                normalized_input_string = ''.join(input_string.split())
                
                is_match = normalized_file_content == normalized_input_string
                if not is_match:
                    with open(file_path, 'w', encoding=self.ENCODING) as out:
                        out.write(", ".join(map(str, input_string)).replace("\\n", "\n"))

            except Exception as e:
                print(f"An error occurred: {e}")
                return False

    def get_prefix_pairs(self, xform_type: XformType) -> List :
        key_prefix_pairs_req_2_pseudo = [
            (self.LITTERAL, "These requirements are for a code file named: {code_fname}"),
            ("requirements", "Use the following requirements to write the pseudocode description:\n"),
            ("architecture", "Use the, following architecture for the write the pseudocode STEP_ACTION_TABLE and `architecture:` section:\n"),
        ]
        key_prefix_pairs_req_2_code = [
            (self.LITTERAL, "Code shall be saved in a file named: {dest_fname}"),
            ("requirements", "Use the following requirements to write code:\n"),
            ("architecture", "Use the following architecture to implement code:\n"),
            ("interface", "Use the following interface implementation requirements:\n"),
            ("error_handling", "Use the following error handling requirements:\n"),
            ("impl_requirements", "Use these additional implementation requirements:\n")
        ]
        key_prefix_pairs_code_to_test = [
            (self.LITTERAL, "Unit Test code shall be saved in a file named: {dest_fname}"),
            (self.LITTERAL, "Target code to be tested comes from a file named: {code_fname}"),
            ("test_requirements", "see the additional test requirements:\n")
        ]
        if xform_type is XformType.PSEUDO:
            return key_prefix_pairs_req_2_pseudo
        elif xform_type == XformType.CODE:
            return key_prefix_pairs_req_2_code
        elif xform_type == XformType.TEST:
            return key_prefix_pairs_code_to_test
        else:
            raise ValueError(f"Invalid target: {xform_type}. Expected one of: Options: {', '.join([t.value for t in XformType])}")

    def process(self, xform_type: XformType, policy_fname: str, source_fname: str, 
                dest_fname: str, code_fname: str) -> None:
        """
        Process the input YAML files to generate a code prompt and save results to specified files.
        Parameters:
        """
        try:
            # EXTRACT RULES - a llm prompt must start with high level system role and user role
            with open(policy_fname, "r", encoding=self.ENCODING) as file:
                data = yaml.safe_load(file)
            prompt = [
                {"role": "system", "content": data["role_system"]},
                {"role": "user", "content": data["role_user"]}
            ]

            # XFORM_TYPE SPECIFIC - pre-processing
            source = source_fname
            target = dest_fname
            if xform_type is XformType.TEST: 
                with open(code_fname, "r", encoding=self.ENCODING) as file:
                    code = file.read()
                prefix = "Write Unit Test for the following code:\n"    
                prompt.append({"role": "user", "content": prefix + code})

            # EXTRACT REQUIREMENTS - from req YAML using `key_prefix_pairs` list 
            key_prefix_pairs = self.get_prefix_pairs(xform_type)
            with open(source, "r", encoding=self.ENCODING) as file:
                arch = yaml.safe_load(file)
                for key, prefix in key_prefix_pairs:
                    if key in arch:
                        content = arch[key]
                        if "{source_fname}" in content:
                            content = content.format(source_fname=source_fname)
                        if "{dest_fname}" in content:
                            content = content.format(dest_fname=dest_fname)
                        if "{code_fname}" in content:
                            content = content.format(code_fname=code_fname)
                        prompt.append({"role": "user", "content": prefix + content})

            # COMPARE PREV_PROMPT w/ NEW_PROMPT
            prompt_fname = Path(dest_fname).with_suffix(".md")
            prompt_fname = str(Path(prompt_fname).parent / f"{self.PROMPT_PREFIX}{Path(prompt_fname).name}")
            is_match = self.compare_file_to_string(prompt_fname, prompt)
            print(f"   {dest_fname} - GPT PROMPTS MATCH" if is_match else f"    {dest_fname} - PROMPT STALE - force regeneration of {dest_fname}")

            # ONLY PROCESS IF FILE if MSGS HAVE CHANGED
            if not is_match:
                response, tokens = self.process_chat(prompt)
                result = f"# TOKENS: {tokens[0] + tokens[1]} (of:{self.max_tokens}) = {tokens[0]} + {tokens[1]}(prompt+return) -- MODEL: {tokens[2]}"
                if response is None:
                    raise RuntimeError("Failed to generate response.")

                # XFORM_TYPE SPECIFIC - post-processing
                if xform_type is XformType.PSEUDO:
                    response = re.sub(r'```yaml', '', response)
                    response = re.sub(r'```.*$', '', response)
                    result +=  '\n' +  response 
                else:
                    result += self.extract_code_from_response(response) 
                with open(target, 'w', encoding=self.ENCODING) as out:
                    out.write(result)

        except Exception as e:
            result = (f"An error occurred while processing files:\n  Input files: {policy_fname}, {source_fname}\n  "
                    f"Output file: {dest_fname}\n  Error details: {e}")
            print(f"ERROR THROWN {result}")
            raise

def main():
    # install_and_import("openai")
    # install_and_import("yaml")
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_MODEL = 'gpt-4o'     # Speed: 17/12/7 seconds
    # DEFAULT_MODEL = 'gpt-3.5-turbo'  # Speed: 5/4/4 seconds

    decl_fname = ""
    result = None
    e_msg = "\nFailed to generate response."
    parser = ArgumentParser()
    args = parser.parse()
    # if args is None:
    #     print("Error: Failed to parse arguments.")
    #     return

    # Access the arguments and use them as needed
    # print(f"Sources: {args.sources}  Destination: {args.destination} Temp: {args.temperature}, rules:{args.rules}")

    try:
      # client = LlmClient(temperature=args.temperature, max_tokens=args.maxtokens, model=args.model)
        client = LlmClient(temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS, model=DEFAULT_MODEL)

        req_fname = "counter_req.yaml"
        pseudo_fname = "counter_pseudo.yaml"
        test_fname = "counter_test.py"
        code_fname = "counter_code.py"

        # 4. **High-Level to Low-Level Requirements Translation**
        policy_fname = Path(LlmClient.WORKFLOW_DIR) / LlmClient.POLICY_DIR / "policy_pseudo.yaml"        
        xform_type   = XformType.PSEUDO
        source_fname = req_fname
        dest_fname   = pseudo_fname
        client.process(xform_type, policy_fname, source_fname, dest_fname, code_fname )
        # client.process( xform_type, rules_fname=args.rules, source_fname=args.source,
        #     dest_fname=args.dest, code_fname=args.code, )

        # 7. **Low-Level Requirements to Code Translation**
        policy_fname = Path(LlmClient.WORKFLOW_DIR) / LlmClient.POLICY_DIR / "policy_python3.8.yaml"        
        xform_type = XformType.CODE
        source_fname = pseudo_fname
        dest_fname   = code_fname
        client.process(xform_type, policy_fname, source_fname, dest_fname, code_fname)

        # 10. **Code to Unit Test Translation**
        policy_fname = Path(LlmClient.WORKFLOW_DIR) / LlmClient.POLICY_DIR / "policy_pytest.yaml"        
        xform_type = XformType.TEST
        source_fname = pseudo_fname
        dest_fname   = test_fname
        client.process(xform_type, policy_fname, source_fname, dest_fname, code_fname)

    except Exception as e:
        e_msg = f"An error occurred while processing files:\n  Input files: {policy_fname}\n  Output file: {code_fname}\n  Error details: {e}"
        print(f"ERROR THROWN {e_msg}")

 
if __name__ == "__main__":
    main()
