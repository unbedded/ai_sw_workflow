## TODO - if only test req was edited - then only remake test
## TODO - if xform.py is newer than _req.yaml  - warn user if
## TODO - why do these fail? 
             # ("/home/preact/Music/240110_13WPM.wav", 20, 10, 20, 13),
             # ("/home/preact/Music/220112_10WPM.wav", 20, 10, 20, 10),


import os
import re
import yaml
import openai
from datetime import datetime
from argument_parser.argument_parser import ArgumentParser
from pathlib import Path

class LlmClient:
    """
    TODO: need to add GPT error checking and handling
    """
    ENCODING = 'utf-8'
    MODEL_KEY_ENV_VARIABLE = "OPENAI_API_KEY"
    PROMPT_PREFIX = ".prompt_"

    def __init__(self, max_tokens:int, temperature:float, model:str):
        api_key = os.getenv(self.MODEL_KEY_ENV_VARIABLE)
        self.client = openai.OpenAI(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.req_tokens = 0
        self.model = model

    def process_chat(self, messages: list) -> str:
        try:
            # Create a chat completion
            response = self.client.chat.completions.create(
                messages=messages,
                max_tokens = self.max_tokens,
                temperature = self.temperature,
                model=self.model
            )

            # Accessing usage field if the response is an object
            self.req_tokens = 0
            # Accessing usage directly if 'usage' is an object
            if hasattr(response, 'usage'):
                prompt_tokens       = response.usage.prompt_tokens  # Directly access the attribute
                completion_tokens   = response.usage.completion_tokens  # Directly access the attribute
                total_tokens        = response.usage.total_tokens  # Directly access the attribute
                print(f"\t\t TOKENS: {total_tokens} (of:{self.max_tokens}) = {prompt_tokens} + {completion_tokens} (prompt+return) - {response.model}")
            else:
                print("Usage information is not available in the response.")            
            self.req_tokens =  response.usage.total_tokens
            if self.req_tokens >=  self.max_tokens:
                print(f"\n\n WARNING:  Max tokens limit reached. {self.max_tokens} tokens.\n\n")

            return response.choices[0].message.content

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


    def extract_code_from_response(self, response: str) -> str:
        # remove everything from the start of the string up to and including the first triple quote
        response = re.sub(r'^.*?\`\`\`', '', response, flags=re.DOTALL)
        # remove everything from the last occurrence of triple quotes to the end of the string
        response = re.sub(r'```.*', '', response, flags=re.DOTALL)
        # Use regex to replace triple quotes at the beginning of any line with '#'
        response = re.sub(r'^([\`]{3})', '#', response, flags=re.MULTILINE)
        # Use regex to replace 'python' beginning of any line with '#'
        response = re.sub(r'^python', '#', response, flags=re.MULTILINE)
        return response


    def compare_file_to_string(self, file_path: str, input_string: str) -> bool:
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
            with open(file_path, 'r') as file:
                file_content = file.read()
            
            normalized_file_content = ''.join(file_content.split())
            normalized_input_string = ''.join(input_string.split())
            
            return normalized_file_content == normalized_input_string
        except Exception as e:
            print(f"An error occurred: {e}")
            return False


    def process(self, code_fname: str, rules_fname: str, 
                uc_fname: str, req_fname: str,
                test_fname:str ) -> None:
        content = ""
        disabled = False
        # READ REQUIREMENTS
        with open(req_fname, "r") as file:
            arch = yaml.safe_load(file)
        target = arch.get("target")
        architecture = arch.get("architecture")
        interface = arch.get("interface")
        requirements = arch.get("requirements")
        error_handling      = arch.get("error_handling")
        test_requirements   = arch.get("test_requirements")
        impl_requirements   = arch.get("impl_requirements")
        imports             = arch.get("imports")
        uc_disabled         = arch.get("uc_disabled")
        test_disabled         = arch.get("test_disabled")
    
        # READ RULES - Load the YAML content from a file
        with open(rules_fname, "r") as file:
            data = yaml.safe_load(file)
        messages=[
            {"role": "system", "content": data["role_system"]},
            {"role": "user", "content": data["role_user"] }
        ]

        # REQUIREMENTS --> USE CASE
        if "usecase" in rules_fname:
            dest_fname = uc_fname
            if uc_disabled is True:
                disabled = True
                content = "UC disabled"
                print(f"\t\t USECASE DISABLED")
            else:
                if requirements:
                    cmd = "Use the following requirements to create a use case:\n"
                    messages.append({"role": "user", "content": cmd + requirements}) 
                    print(f"\t\t REQUIREMENTS: {requirements}")   
                else:
                    raise KeyError("The required element 'requirements' was not found in the YAML file.")
                # Jump to PROCESS

        # USE CASE --> CODE
        elif "python" in rules_fname:
            dest_fname = code_fname
            if uc_disabled is True:
                if requirements:
                    # Instead of a UC - use these requirements to write code
                    cmd = "Use the following requirements to write code:\n"
                    messages.append({"role": "user", "content": cmd + requirements})    

            name = code_fname.split("/")
            if len(name) > 1:
                dir = name[0]           
                fname = name[1]
            else:
                dir = ""           
                fname = name[0]
            cmd = f"Note that generted code will be stored in a file named {fname} in a directory named {dir}. Also document this fact in the header:\n"
            messages.append({"role": "user", "content": cmd})
            if architecture:
                cmd = "Use the following architecture to implement code:\n"
                messages.append({"role": "user", "content": cmd + architecture})
            if interface:
                cmd = "Use the following interface implementation requirements:\n"
                messages.append({"role": "user", "content": cmd + interface})
            if error_handling:
                cmd = "Use the following error handling requirements:\n"
                messages.append({"role": "user", "content": cmd + error_handling})
            if uc_disabled is False:
                with open(uc_fname, "r") as file:
                    usecase = file.read()
                cmd = "Implement requirements defined in the following use case:\n"
                messages.append({"role": "user", "content": cmd + usecase})
            if impl_requirements:
                cmd = "Use these additional implementation requirements:\n"
                messages.append({"role": "user", "content": cmd + impl_requirements})
            if imports:
                imports = imports.replace(" ", "").replace("-", "").split("\n")
                imports = [item for item in imports if item != '']
                for imp in imports:
                    # IDEALLY - would be good to have code name and subdirectory for the module declaration
                    mod_dir = imp.split("/")[0]
                    fname   = imp.split("/")[1]
                    decl_fname  = imp.replace("_xform.py", "_decl.md")
                    module  = fname.replace(".py", "")
                    cmd =  f"Use the following as instructions to understand how to use the module: {module}.\n"
                    cmd += f"The module is stored in a file named {name} from the subdirectory {mod_dir} :\n"
                    cmd += f"Use the provided code exactly as it is. Do not modify method names, signatures, or any class behavior.\n"
                    cmd += f"Do not make assumptions about the module: {module}.\n" 
                    cmd += f"Only use the methods and classes exactly as specified in the provided code."
                    with open(decl_fname, 'r', encoding=self.ENCODING) as file:
                        decl = file.read()
                    messages.append({"role": "user", "content": cmd + decl})
                # Jump to PROCESS

        # CODE --> TEST
        elif "ptest" in rules_fname:
            dest_fname = test_fname
            if test_disabled is True:
                disabled = True
                content = "# Unit Testing disabled"
                print(f"\t\t UNIT TEST DISABLED")
            else:
                # READ UseCase  - Stripping out comments
                with open(uc_fname, "r", encoding=self.ENCODING) as file:
                    usecase = ''.join(line if not line.strip().startswith('##') else '' for line in file)
                    # usecase = file.read()
                with open(code_fname, 'r', encoding=self.ENCODING) as file:
                    code = ''.join(line if not line.strip().startswith('##') else '' for line in file)
                    # code = file.read()
                cmd = "Write unit tests must verify the following python code. This code comes from a file named {code_fname} :\n"
                messages.append({"role": "user", "content": cmd + code})
                cmd = "Ensure the unit test verify the following requirements:\n"
                messages.append({"role": "user", "content": cmd + usecase})
                if test_requirements:
                    cmd = "Use the additional test requirements:\n"
                    messages.append({"role": "user", "content": cmd + test_requirements})
                # Jump to PROCESS

        # CODE --> DECLARATION
        elif "decl" in rules_fname:
            decl_fname = code_fname.replace("_xform.py", "_decl.md")
            dest_fname = decl_fname
            if test_disabled is True:
                disabled = True
                content = "Unit Testing disabled"
            else:
                # READ CODE 
                with open(code_fname, "r", encoding=self.ENCODING) as file:
                    code = ''.join(line if not line.strip().startswith('##') else '' for line in file)
                    # code = file.read()
                mod_dir = code_fname.split("/")[0]
                name = code_fname.split("/")[1]
                cmd = f"The following code comes from a file named {name} in a subdirectory named {mod_dir} :\n"
                messages.append({"role": "user", "content": cmd + code})
                # Jump to PROCESS

        else: # DEFAULT - ERROR HANDLER
            msg = f"Error: Invalid dest file name. {dest_fname}"
            print(msg)
            content = msg

        # ONLY PROCESS IF REQUIRED
        is_match = False
        if disabled is False:
            prompt = ", ".join(map(str, messages))
            prompt = prompt.replace("\\n", "\n")
            prompt_fname = Path(dest_fname).with_suffix(".md")   
            new_prefix = "/" + self.PROMPT_PREFIX
            prompt_fname = str(prompt_fname).replace("/", new_prefix )   

            # CHECK IF MSG HAS CHANGED
            is_match = self.compare_file_to_string(prompt_fname, prompt)
            print(f"\t\t GPT PROMPTS MATCH" if is_match else f"\t\t PROMPT STALE - force regeneration of {dest_fname}")

            # ONLY PROCESS IF FILE if MSGS HAVE CHANGED
            if not is_match:
                content = self.process_chat(messages)
                if dest_fname.endswith(".py"):
                    content = self.extract_code_from_response(content)
                # print(f"\t\t SAVING PROMPT TO: {prompt_fname}")
                with open(prompt_fname, 'w', encoding=self.ENCODING) as file:
                    file.write(prompt)

        # SAVE TO DESTINATION 
        if not is_match:
            current_datetime: datetime = datetime.now()
            formatted_datetime: str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            timestamp: str = f"## Creation Timestamp : {formatted_datetime}\n"
            token_usage = f"## Token Usage: {self.req_tokens} of {self.max_tokens} tokens\n"
            if self.req_tokens >= self.max_tokens:
                content = f"\n\n  WARNING  Max tokens limit reached.\n{self.max_tokens} tokens."

            print(f"\t\t SAVING TO: {dest_fname} - # of lines:{len(content.splitlines())}")
            with open(dest_fname, 'w', encoding=self.ENCODING) as file:
                file.write(timestamp)
                file.write(token_usage) 
                file.write(content)



def main():
    decl_fname = ""
    parser = ArgumentParser()
    args = parser.parse()
    if args is None:
        print("Error: Failed to parse arguments.")
        return
    # Access the arguments and use them as needed
    # print(f"Sources: {args.sources}  Destination: {args.destination} Temp: {args.temperature}, rules:{args.rules}")
    client = LlmClient(temperature=args.temperature, max_tokens=args.maxtokens, model=args.model)
    client.process(code_fname=args.code, rules_fname=args.rules,
                   uc_fname=args.usecase, req_fname=args.requirements,
                   test_fname=args.test)
 
if __name__ == "__main__":
    main()
