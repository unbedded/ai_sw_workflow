# ┌──────────────┐     ┌─────────────┐    ┌─────────────┐    ┌────────────┐
# │ Requirements │     │ Use Case    │    │ Class/Func  │    │ PyTest     │
# │              ┼────►│             ┼───►│             ┼───►│            │
# │  *_req.yaml  │ ▲   │   *_uc.md   │ ▲  │  *_xform.py │ ▲  │ *_test.py  │
# └──────────────┘ │   └─────────────┘ │  └─────────────┘ │  └────────────┘
#                  │                   │                  │                
#               rules_uc.md     rules_python.md      rules_pytest.md     
#                                                             
############################################################################

# Variables
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
MODEL = 'gpt-4o'
# MODEL = 'gpt-4o-mini'
# MODEL = 'o1-mini'
# MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 8000
TEMPERATURE = 0.1
MAIN_SCRIPT = dev_gpt_main.py -m=$(MAX_TOKENS) -T=$(TEMPERATURE) --model=$(MODEL)

# Define source and destination suffixes
REQ_SUFFIX  = _req.yaml
UC_SUFFIX   = _uc.md
PY_SUFFIX   = _xform.py
PTEST_SUFFIX = _test.py
DECL_SUFFIX = _decl.md

# Define Rule File names
RULE_DIR  = ./rules/
RULE_UC   = $(RULE_DIR)rules_usecase.yaml
RULE_PY   = $(RULE_DIR)rules_python.yaml
RULE_PTEST= $(RULE_DIR)rules_ptest.yaml
RULE_DECL = $(RULE_DIR)rules_decl.yaml

# Find all source files in subdirectories with the specified postfixes
EXCLUDE_SOURCES = \( -name "template_req.yaml" -o -name "exclude_this.yaml" \)
REQ_SOURCES = $(shell find . -depth -mindepth 2 -type f -name "*$(REQ_SUFFIX)" -not $(EXCLUDE_SOURCES))
DECL_SOURCES =$(shell find . -depth -mindepth 2 -type f -name "*$(REQ_SUFFIX)" -not $(EXCLUDE_SOURCES))

# Generate corresponding destination file names
UC_DESTINATIONS    = $(REQ_SOURCES:$(REQ_SUFFIX)=$(UC_SUFFIX))
PY_DESTINATIONS    = $(REQ_SOURCES:$(REQ_SUFFIX)=$(PY_SUFFIX))
PTEST_DESTINATIONS = $(REQ_SOURCES:$(REQ_SUFFIX)=$(PTEST_SUFFIX))
DECL_DESTINATIONS  =$(DECL_SOURCES:$(REQ_SUFFIX)=$(DECL_SUFFIX))

# Combine all destinations
DESTINATIONS = $(UC_DESTINATIONS) $(PY_DESTINATIONS) $(PTEST_DESTINATIONS) $(DECL_DESTINATIONS)
.PRECIOUS: $(DESTINATIONS)

.PHONY: setup run clean help

all: $(PTEST_DESTINATIONS)  $(DECL_DESTINATIONS) count_lines

# Setup virtual environment if it does not already exist
setup:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Setting up virtual environment..."; \
		python3.13 -m venv $(VENV_DIR); \
		$(PIP) install -r requirements.txt; \
	else \
		echo "Virtual environment already exists, skipping setup."; \
	fi

count_lines:
	@echo "Counting lines of Python code and YAML files..."
	@python_lines=$$(find . -type f -name "*_xform.py" ! -name "*_test.py" ! -path "*/__pycache__/*" ! -path "./.cache/*" ! -path "./.git/*" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \
	test_lines=$$(find . -type f -name "*_test.py" ! -path "*/__pycache__/*" ! -path "./.cache/*" ! -path "./.git/*" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \
	yaml_lines=$$(find . -type f -name "*.yaml" ! -path "./rules/*" ! -path "./.cache/*" ! -path "./.git/*" -exec wc -l {} + | awk '{total += $$1} END {print total}'); \	total_lines=$$(($$python_lines + $$test_lines + $$yaml_lines)); \
	total_lines=$$(($$python_lines + $$test_lines + $$yaml_lines)); \
	echo "TOTAL LINES: $$total_lines = Code: $$python_lines + Test: $$test_lines (YAML: $$yaml_lines)"

# Rule to generate _uc.md from _req.md
%$(UC_SUFFIX): %$(REQ_SUFFIX)
	$(PYTHON) $(MAIN_SCRIPT)  --rules $(RULE_UC) --requirements $*$(REQ_SUFFIX) --usecase $@ 

# Rule to generate .py from _uc.md
%$(PY_SUFFIX): %$(UC_SUFFIX) %$(REQ_SUFFIX)
	$(PYTHON) $(MAIN_SCRIPT)  --rules $(RULE_PY) --requirements $*$(REQ_SUFFIX) --usecase $< --code $@

# Rule to generate ptest.py from _xform.py
%$(PTEST_SUFFIX): %$(PY_SUFFIX) %$(UC_SUFFIX)
	$(PYTHON) $(MAIN_SCRIPT) --rules $(RULE_PTEST) --requirements $*$(REQ_SUFFIX) --usecase $*$(UC_SUFFIX) --code $< --test $@   

# Rule to generate _decl.md from _xform.py
%$(DECL_SUFFIX): %$(PY_SUFFIX)
	$(PYTHON) $(MAIN_SCRIPT) --rules $(RULE_DECL) --requirements $*$(REQ_SUFFIX) --code $<   

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
	@echo "  setup     - Create a virtual environment and install dependencies"
	@echo "  run       - Find source files, check for stale destinations, and update them"
	@echo "  clean     - Remove generated files and clean the environment"
	@echo "  help      - Display this help message"
