# ┌──────────────┐     ┌─────────────┐    ┌─────────────┐    ┌────────────┐
# │ Requirements │     │ Use Case    │    │ Class/Func  │    │ PyTest     │
# │              ┼────►│             ┼───►│             ┼───►│            │
# │  *_req.yaml  │ ▲   │   *_uc.md   │ ▲  │  *_xform.py │ ▲  │ *_test.py  │
# └──────────────┘ │   └─────────────┘ │  └─────────────┘ │  └────────────┘
#                  │                   │                  │                
#               policy_uc.md     policy_python.md      policy_pytest.md     
#                                                             
############################################################################

# Variables
VENV_DIR = .venv
WORKFLOW_DIR = ai_sw_workflow
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
MODEL = 'gpt-4o'
# MODEL = 'gpt-4o-mini'
# MODEL = 'o1-mini'
# MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 8000
TEMPERATURE = 0.1
MAIN_SCRIPT = $(WORKFLOW_DIR)/ai_sw_workflow.py -m=$(MAX_TOKENS) -T=$(TEMPERATURE) --model=$(MODEL)

# Define source and destination suffixes
REQ_SUFFIX  = _req.yaml
PSEUDO_SUFFIX  = _pseudo.yaml
PCODE_SUFFIX   = _code.py
PTEST_SUFFIX   = _test.py

# Define Rule File names
POLICY_DIR  = ./$(WORKFLOW_DIR)/policy
POLICY_PSEUDO=$(POLICY_DIR)/policy_pseudo.yaml
POLICY_PY38 = $(POLICY_DIR)/policy_python3.8.yaml
POLICY_PTEST= $(POLICY_DIR)/policy_pytest.yaml

# Find all source files in subdirectories with the specified postfixes
EXCLUDE_SOURCES = -path "./$(WORKFLOW_DIR)/*" -prune
REQ_SOURCES = $(shell find . -depth -mindepth 1 -type f -name "*$(REQ_SUFFIX)" -not \( $(EXCLUDE_SOURCES) \))

# Generate corresponding destination file names
PSEUDO_DESTINATIONS = $(REQ_SOURCES:$(REQ_SUFFIX)=$(PSEUDO_SUFFIX))
PCODE_DESTINATIONS  = $(REQ_SOURCES:$(REQ_SUFFIX)=$(PCODE_SUFFIX))
PTEST_DESTINATIONS  = $(REQ_SOURCES:$(REQ_SUFFIX)=$(PTEST_SUFFIX))

# Combine all destinations
DESTINATIONS = $(PCODE_DESTINATIONS) $(PTEST_DESTINATIONS) $(PSEUDO_DESTINATIONS)
.PRECIOUS: $(DESTINATIONS)

# .PHONY: setup run clean help
.PHONY:  run clean help

all: $(PTEST_DESTINATIONS)  count_lines
# all: $(PTEST_DESTINATIONS) $(PSEUDO_DESTINATIONS) count_lines

# Setup virtual environment if it does not already exist
# setup:
# 	@if [ ! -d "$(VENV_DIR)" ]; then \
# 		echo "Setting up virtual environment..."; \
# 		python3.13 -m venv $(VENV_DIR); \
# 		$(PIP) install -r requirements.txt; \
# 	else \
# 		echo "Virtual environment already exists, skipping setup."; \
# 	fi

count_lines:
	@echo "Counting lines of Python code and YAML files..."
	@python_lines=$$(find . -type f -name "*_xform.py" ! -name "*_test.py" ! -path "*/__pycache__/*" ! -path "./.cache/*" ! -path "./.git/*" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \
	test_lines=$$(find . -type f -name "*_test.py" ! -path "*/__pycache__/*" ! -path "./.cache/*" ! -path "./.git/*" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \
	yaml_lines=$$(find . -type f -name "*.yaml" ! -path "./policy/*" ! -path "./.cache/*" ! -path "./.git/*" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \	total_lines=$$(($$python_lines + $$test_lines + $$yaml_lines)); \
	total_lines=$$(($$python_lines + $$test_lines + $$yaml_lines)); \
	echo "TOTAL LINES: $$total_lines = Code: $$python_lines + Test: $$test_lines (YAML: $$yaml_lines)"

## RULES first prerequisite ($<)
## RULES second prerequisite ($^)
## RULES third prerequisite ($?)
## RULES target ($@)

# Rule to generate _uc.md from _req.md
%$(PSEUDO_SUFFIX): %$(REQ_SUFFIX)
	@$(PYTHON) $(MAIN_SCRIPT) --source $< --dest $@  --xform pseudo --policy $(POLICY_PSEUDO) --code "n.a."

# Rule to generate .py from _uc.md
%$(PCODE_SUFFIX): %$(PSEUDO_SUFFIX)
	@$(PYTHON) $(MAIN_SCRIPT) --source $< --dest $@  --xform code   --policy $(POLICY_PY38)  --code $@

# Rule to generate ptest.py from _xform.py
%$(PTEST_SUFFIX): %$(PCODE_SUFFIX) %$(PSEUDO_SUFFIX)
	@$(PYTHON) $(MAIN_SCRIPT) --source $(word 2,$^) --dest $@  --xform test  --policy $(POLICY_PTEST) --code $<    

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf  $(DESTINATIONS)
	find . -type f -name ".prompt_*" -exec rm -f {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Display help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Set up the environment and run the script if needed"
	# @echo "  setup     - Create a virtual environment and install dependencies"
	@echo "  run       - Find source files, check for stale destinations, and update them"
	@echo "  clean     - Remove generated files and clean the environment"
	@echo "  help      - Display this help message"
