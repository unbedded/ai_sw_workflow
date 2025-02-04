# ┌──────────────┐     ┌─────────────┐    ┌─────────────┐    ┌────────────┐
# │ Requirements │     │ LowLvl Req  │    │ Class/Func  │    │ PyTest     │
# │              ┼────►│             ┼───►│             ┼───►│            │
# |*_recipe.yaml │ ▲   │ *_pseudo.md │ ▲  │  *_code.py  │ ▲  │ *_test.py  │
# └──────────────┘ │   └─────────────┘ │  └─────────────┘ │  └────────────┘
#                  │                   │                  │                
#               policy_pseudo.md  policy_code.md      policy_test.md     
#                                                             
############################################################################

# Set default parallel jobs 
NPROC := 2
XPROC := $(shell echo $$(( $(shell nproc) < $(NPROC) ? $(shell nproc) : $(NPROC) )))
MAKEFLAGS += -j$(XPROC)

# Variables
VENV_DIR = .venv
WORKFLOW_DIR = ai_sw_workflow
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
MODEL = 'gpt-4o'
MAX_TOKENS = 8000
TEMPERATURE = 0.1
MAIN_SCRIPT = $(WORKFLOW_DIR)/ai_sw_workflow.py -m=$(MAX_TOKENS) -t=$(TEMPERATURE) --model=$(MODEL)

# C++ Compiler and flags
GTEST_INCLUDE = /usr/local/include/gtest
GTEST_LIB = /usr/local/lib
CXX = g++
CXXFLAGS = -std=c++2a -I$(GTEST_INCLUDE) 
LDFLAGS =  -lgtest -lgtest_main -pthread

# Define source and destination suffixes
RECIPE_SUFFIX  = _recipe.yaml
PSEUDO_SUFFIX  = _pseudo.yaml
POLICY_DIR  = ./$(WORKFLOW_DIR)/policy
POLICY_PSEUDO=$(POLICY_DIR)/policy_pseudo.yaml

# Conditional variable to switch policies
# POLICY_MODE = c++20
POLICY_MODE = python3.8

ifeq ($(POLICY_MODE), python3.8)
CODE_SUFFIX   = _code.py
TEST_SUFFIX   = _test.py
POLICY_CODE = $(POLICY_DIR)/policy_python3.8.yaml
POLICY_TEST= $(POLICY_DIR)/policy_pytest.yaml
else ifeq ($(POLICY_MODE), c++20)
CODE_SUFFIX   = _code.cpp
TEST_SUFFIX   = _test.cpp
POLICY_CODE = $(POLICY_DIR)/policy_c++20.yaml
POLICY_TEST= $(POLICY_DIR)/policy_gtest.yaml
endif

# Find all source files up to 4 levels deep (excluding workflow directory)
RECIPE_SOURCES = $(shell find . -mindepth 1 -maxdepth 3 -type f -name "*$(RECIPE_SUFFIX)" ! -path "./$(WORKFLOW_DIR)/*")

# Generate corresponding destination file names dynamically
PSEUDO_DESTINATIONS = $(RECIPE_SOURCES:$(RECIPE_SUFFIX)=$(PSEUDO_SUFFIX))
CODE_DESTINATIONS   = $(RECIPE_SOURCES:$(RECIPE_SUFFIX)=$(CODE_SUFFIX))
TEST_DESTINATIONS   = $(RECIPE_SOURCES:$(RECIPE_SUFFIX)=$(TEST_SUFFIX))

# Combine all destinations
DESTINATIONS = $(PSEUDO_DESTINATIONS) $(CODE_DESTINATIONS) $(TEST_DESTINATIONS)

.PRECIOUS: $(DESTINATIONS)
.PHONY: all process_files clean test count_lines template rm_cfg

# Entry point
all: process_files rm_cfg

# Process files in depth-first order
process_files: $(DESTINATIONS)

# Depth-first processing rules
# %$(PSEUDO_SUFFIX): %$(RECIPE_SUFFIX)
# #	@echo "Generating: $@ from $<"
# 	@$(PYTHON) $(MAIN_SCRIPT) --recipe $< --dest $@  --xform pseudo --policy $(POLICY_PSEUDO) --code "n.a."

# %$(CODE_SUFFIX): %$(PSEUDO_SUFFIX) %$(RECIPE_SUFFIX)
%$(CODE_SUFFIX):  %$(RECIPE_SUFFIX)
#	@echo "Generating: $@ from $<"
	@$(PYTHON) $(MAIN_SCRIPT) --recipe $< --dest $@  --xform code --policy $(POLICY_CODE) --code $@

%$(TEST_SUFFIX): %$(CODE_SUFFIX) %$(RECIPE_SUFFIX)
#	@echo "Generating: $@ from $<"
	@$(PYTHON) $(MAIN_SCRIPT) --recipe $(word 2,$^) --dest $@ --xform test --policy $(POLICY_TEST) --code $<

ifeq ($(POLICY_MODE), c++20)
	$(CXX) $(CXXFLAGS) -L$(GTEST_LIB) $@ $< $(LDFLAGS) -o $(@:.cpp=)
	$(@:.cpp=)
endif

# Test execution
test:
ifeq ($(POLICY_MODE), python3.8)
	coverage run -m pytest --tb=line | grep -vE "^(platform|rootdir|plugins|collected)"
	coverage report -m
else ifeq ($(POLICY_MODE), c++20)
	@for test in $(GTEST_EXECUTABLES); do \
		echo "Running $$test"; \
		$$test; \
	done
endif

# Count lines of Python and YAML code
count_lines:
	@echo "Counting lines of Python code and YAML files..."
	@python_lines=$$(find . -type f -name "*_xform.py" ! -name "*_test.py" ! -path "*/__pycache__/*" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \
	test_lines=$$(find . -type f -name "*_test.py" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \
	yaml_lines=$$(find . -type f -name "*.yaml" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \
	total_lines=$$(($$python_lines + $$test_lines + $$yaml_lines)); \
	echo "TOTAL LINES: $$total_lines = Code: $$python_lines + Test: $$test_lines + YAML: $$yaml_lines"

# Template generation
template:
	@if [ "$(name)" = "" ]; then \
		echo "Error: Please provide a name variable. Example: make template name=my_template"; \
		exit 1; \
	fi
	cp -r ai_sw_workflow/template ./$(name)
	mv ./$(name)/template_recipe.yaml ./$(name)/$(name)_recipe.yaml

# Cleanup rules
rm_cfg:
	rm -f morse_cfg.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf $(DESTINATIONS)
	find . -type f -name ".prompt_*" -exec rm -f {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name '*.out' -exec rm -f {} +	
	find . -type f -name '*_code.hpp' -delete
	find . -type f -name '*.log' -delete
	rm_cfg

# Display help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Process files and clean configuration"
	@echo "  process_files - Process all required files recursively"
	@echo "  test      - Run tests based on selected POLICY_MODE"
	@echo "  clean     - Remove generated files and clean environment"
	@echo "  template name=<name> - Create a new template"

