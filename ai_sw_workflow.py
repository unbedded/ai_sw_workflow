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
from argument_parser import ArgumentParser, XformType
from pathlib import Path
from enum import Enum

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

class LlmClient:
    WORKFLOW_DIR = "ai_sw_workflow"
    POLICY_DIR = "policy"
    ENCODING = 'utf-8'
    PROMPT_PREFIX = ".prompt_"
    LITTERAL = "LITERAL"
    CODE_REF = "code_references"
    TEST_ENABLE="test_enable"

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
        while self.running:
            seconds += 1
            print(f"\r\t{seconds:>3} {'.' * seconds}", end="", flush=True)
            time.sleep(1)

    def _process_chat(self, messages: List[dict]) -> Tuple[Optional[str], Tuple[int, int, str]]:
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

    def _extract_code_from_response(self, response: str) -> str:
        """Extract and clean code from a response string."""
        response = re.sub(r'^.*?\`\`\`', '', response, flags=re.DOTALL)
        response = re.sub(r'```.*', '', response, flags=re.DOTALL)
        response = re.sub(r'^([\`]{3})', '#', response, flags=re.MULTILINE)
        response = re.sub(r'^python', '#', response, flags=re.MULTILINE)
        return response

    def _prompt_compare_with_save(self, dest_fname: str, prompt: List[dict]) -> bool:
        """
        Compares the content of a file to a given string, ignoring all whitespace differences.
        If `dest_fname` file does not exist - then False is returned

        :param file_path: Path to the file to compare.
        :param input_string: The string to compare with the file content.
        :return: True if the file content matches the string ignoring whitespace; False otherwise.
        """
        try:
            prompt_fname = Path(dest_fname).with_suffix(".md")
            prompt_fname = str(Path(prompt_fname).parent / f"{self.PROMPT_PREFIX}{Path(prompt_fname).name}")
            prompt_text = []
            for item in prompt:
                role = item["role"].capitalize()
                content = "\n".join(line for line in item["content"].strip().splitlines() if line.strip())
                prompt_text.append(f"### {role}\n\n{content}\n")
            prompt_text = "\n".join(prompt_text)

            is_match = False
            is_exist = os.path.exists(dest_fname) # True if file exists
            if is_exist:
                if os.path.exists(prompt_fname):
                    with open(prompt_fname, 'r', encoding=self.ENCODING) as file:
                        file_content = file.read()
                    normalized_file_content = ' '.join(file_content.split())
                    normalized_input_string = ' '.join(prompt_text.split())
                    is_match = normalized_file_content == normalized_input_string
            if not is_match:
                with open(prompt_fname, 'w', encoding=self.ENCODING) as out:
                    # out.write(", ".join(map(str, prompt_text)).replace("\\n", "\n"))
                    out.write(prompt_text)
            return is_match        

        except Exception as e:
            print(f"An error occurred: {e}")
            return False


    def process(self, xform_type: XformType, policy_fname: str, source_fname: str, 
                dest_fname: str, code_fname: str) -> None:
        """
        Process the input YAML files to generate a code prompt and save results to specified files.
        Parameters:
        """
        IMPORTS = "imports"
        source = source_fname
        target = dest_fname
        try:            
            # ABORT if TEST_ENABLE is False & xform_type is TEST
            with open(source, "r", encoding=self.ENCODING) as file:
                rules = yaml.safe_load(file)
                enable_test = "True" == rules.get(self.TEST_ENABLE, "False")
                if not enable_test and xform_type is XformType.TEST:
                    print(f"Skipping TEST generation for {source} as {self.TEST_ENABLE} is not set to True.")
                    return  # WARNING - RETURNING EARLY

            # EXTRACT POLICY - a llm prompt must start with high level system role and user role
            key_prefix_pairs = [("key", "prefix")]
            with open(policy_fname, "r", encoding=self.ENCODING) as file:
                policy = yaml.safe_load(file)
                key_prefix_pairs = policy["key_prefix_pairs"]
                prompt = [
                    {"role": "system", "content": policy["role_system"]},
                    {"role": "user", "content": policy["role_user"]}
                ]

            # PRE-PROCESSING
            if xform_type is XformType.TEST: 
                with open(code_fname, "r", encoding=self.ENCODING) as file:
                    code = file.read()
                prefix = f"Write Unit Test for the following code which comes from a file named: {code_fname}:\n"    
                prompt.append({"role": "user", "content": prefix + code})

            # EXTRACT REQUIREMENTS - from req YAML using `key_prefix_pairs` list 
            with open(source, "r", encoding=self.ENCODING) as file:
                rules = yaml.safe_load(file)               
                for key, prefix in key_prefix_pairs:
                    if key == self.LITTERAL:
                        print("TODO - use API string variable sub instead of f-strings")
                        content = prefix
                        if "{source_fname}" in prefix:
                            content = prefix.format(source_fname=source_fname)
                        if "{dest_fname}" in prefix:
                            content = prefix.format(dest_fname=dest_fname)
                        if "{code_fname}" in prefix:
                            content = prefix.format(code_fname=code_fname)
                        prompt.append({"role": "user", "content": content})
                    elif key == self.CODE_REF and key in rules:
                        references_str = rules[key]
                        references = [line.lstrip("- ").strip() for line in references_str.splitlines() if line.strip()]
                        for ref_fname in references:
                            module = ref_fname.split('/')[0]
                            fname  = ref_fname.split('/')[1]
                            cmd = (f"Use the following as instructions to understand how to use the package: {module}.\n"
                                    f"The package's module is stored in a file named {fname} from the subdirectory {module} :\n")
                            with open(ref_fname, 'r', encoding=self.ENCODING) as file:
                                reference = file.read()
                            prompt.append({"role": "user", "content": cmd + reference})
                    elif key in rules and not key.startswith("_"):
                        content = rules[key]
                        prompt.append({"role": "user", "content": prefix + content})

            # COMPARE PREV_PROMPT w/ NEW_PROMPT
            is_match = self._prompt_compare_with_save(dest_fname, prompt)
            print(f"   {dest_fname} - PROMPTS MATCH" if is_match else f"    {dest_fname} - PROMPT STALE - force regeneration")

            # ONLY PROCESS IF FILE if MSGS HAVE CHANGED
            if not is_match:
                response, tokens = self._process_chat(prompt)
                result = f"# TOKENS: {tokens[0] + tokens[1]} (of:{self.max_tokens}) = {tokens[0]} + {tokens[1]}(prompt+return) -- MODEL: {tokens[2]}"
                if response is None:
                    raise RuntimeError("Failed to generate response.")

                # POST-PROCESSING
                if not xform_type is XformType.PSEUDO:
                    result += self._extract_code_from_response(response) 
                else:
                    response = response.replace("```yaml", "   ")
                    response = response.replace("```", "   ")
                    result +=  '\n' +  response 

                    # LITERAL reqirements (which are prefix w/ underscore) 
                    # copy them directly into PSEUDO yaml file
                    with open(source, "r", encoding=self.ENCODING) as file:
                        rules = yaml.safe_load(file)
                        for _key, prefix in key_prefix_pairs:
                            key = _key[1:]
                            if _key.startswith("_") and key in rules:
                                content = rules[key]
                                result += f"\n{key}: |\n"
                                result += "\n".join([f"  {line}" for line in content.splitlines()])
                                result += "\n"

                with open(target, 'w', encoding=self.ENCODING) as out:
                    out.write(result)

        except Exception as e:
            result = (f"An error occurred while processing files:\n  Input files: {policy_fname}, {source_fname}\n  "
                    f"Output file: {dest_fname}\n  Error details: {e}")
            print(f"ERROR THROWN {result}")
            raise

def main():
    install_and_import("openai")
    install_and_import("yaml")
    parser = ArgumentParser()
    args = parser.parse()
    if args is None:
        print("Error: Failed to parse arguments.")
        return
    try:
        client = LlmClient(temperature=args.temperature, max_tokens=args.maxtokens, model=args.model)
        client.process( xform_type=args.xform, policy_fname=args.policy, 
            source_fname=args.source, dest_fname=args.dest, code_fname=args.code, )
    except Exception as e:
        e_msg = f"An error occurred while processing files:\n  Input files: {args.policy}\n  Output file: {args.code}\n  Error details: {e}"
        print(f"ERROR THROWN {e_msg}")

if __name__ == "__main__":
    main()
