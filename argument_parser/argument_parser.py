"""
File: argument_parser.py
Date: 2024-11-07

This module provides an ArgumentParser class to handle command-line arguments.
"""

import argparse
from typing import Optional, Any
from enum import Enum

class XformType(Enum):
    PSEUDO = "pseudo"
    CODE = "code"
    TEST = "test"

class ArgumentParser:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description="Parse command-line arguments.")
        self._setup_arguments()

    def _setup_arguments(self) -> None:
        # One-letter argument name with default value and type
        self._parser.add_argument(
            '-p', '--policy',
            required=True,
            type=str,
            default=" ",
            help="A path to the transformer rules file (default: None)"
        )
        self._parser.add_argument(
            '-s', '--source',
            required=True,
            type=str,
            default=" ",
            help="A path source yaml file (required))"
        )
        self._parser.add_argument(
            '-d', '--dest',
            required=True,
            type=str,
            default=" ",
            help="A path to target destination (required)"
        )
        self._parser.add_argument(
            '-c', '--code',
            required=False,
            type=str,
            default=" ",
            help="A path to code implementation - Used when writting UnitTest (default: None)"
        )
        self._parser.add_argument(
            '-x', '--xform',
            type=self.validate_target_type, 
            required=True, 
            help=f"Transform type. Options: {', '.join([t.value for t in XformType])}"
        )
        self._parser.add_argument(
            '-m', '--maxtokens',
            required=False,
            type=int,
            default=2000,
            help="Maximum tokens limit for GPT-3 API (default: 2000)"
        )
        # Bounds checking for float argument
        self._parser.add_argument(
            '-T', '--temperature',
            required=False,
            type=self._bounded_float,
            default=0.1,
            help="The GPT temperature (default: 0.1, range: 0.0 to 1.0)"
        )
        # Bounds checking for float argument
        self._parser.add_argument(
            '-g', '--model',
            required=False,
            type=str,
            default="gpt-3.5-turbo",
            help="The GPT Model - assumes env variable `OPENAI_API_KEY` is set (default: gpt-3.5-turbo)"
        )

    def _bounded_float(self, value: str) -> float:
        """Check if the float value is within bounds (0.0 to 1.0)."""
        float_value = float(value)
        if 0.0 <= float_value <= 1.0:
            return float_value
        raise argparse.ArgumentTypeError("Temperature must be between 0.0 and 1.0")

    def validate_target_type(self, value: str) -> XformType:
        try:
            return XformType(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value '{value}'. Must be one of: {', '.join([t.value for t in XformType])}.")

    def parse(self) -> Optional[argparse.Namespace]:
        try:
            return self._parser.parse_args()
        except SystemExit:
            return None
