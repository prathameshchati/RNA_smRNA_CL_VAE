from typing import Any, Dict
import yaml
import time
from functools import wraps
from colorama import init, Fore, Style

# logging
from _logger import set_log_file, console_print, console_print_bold, save_log

# printing
# def console_out(message):
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# formatted print
# colorama init
# init(autoreset=True)

# def console_print(message, bold=False):
#     timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#     if bold:
#         print(f"{Style.BRIGHT}[{timestamp}] {message}{Style.RESET_ALL}")
#     else:
#         print(f"[{timestamp}] {message}")

# def console_print_bold(message):
#     console_print(message, bold=True)

# load config
def load_config(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

# nested dict for storing config file
def parse_config(yaml_data):
    return {section: dict(content) for section, content in yaml_data.items()}

# formatted print config
def print_config(config: Dict[str, Dict[str, Any]], indent: int = 0) -> None:
    for section, content in config.items():
        console_print(f"{'  ' * indent}{section}:")
        for key, value in content.items():
            if isinstance(value, dict):
                console_print(f"{'  ' * (indent + 1)}{key}:")
                print_config({key: value}, indent + 2)
            elif isinstance(value, list):
                console_print(f"{'  ' * (indent + 1)}{key}:")
                for item in value:
                    console_print(f"{'  ' * (indent + 2)}- {item}")
            else:
                console_print(f"{'  ' * (indent + 1)}{key}: {value}")

# time functions
def block_timer(description):
    return BlockTimer(description)

class BlockTimer:
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        console_print(f"{self.description} executed in {self.execution_time:.4f} seconds")
        print("\n")



