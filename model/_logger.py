import time
from colorama import init, Style
import sys
import io

# Initialize colorama
init(autoreset=True)

class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.log_buffer = []

    def print(self, message, bold=False):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        
        if bold:
            print(f"{Style.BRIGHT}{formatted_message}{Style.RESET_ALL}")
        else:
            print(formatted_message)
        
        # Store the message without ANSI codes
        self.log_buffer.append(formatted_message)

    def print_bold(self, message):
        self.print(message, bold=True)

    def save_log(self):
        if self.log_file:
            with open(self.log_file, 'w') as f:
                for line in self.log_buffer:
                    f.write(line + '\n')
            print(f"Log saved to {self.log_file}")
            self.log_buffer.clear()

# Global logger instance
logger = Logger()

def set_log_file(file_path):
    global logger
    logger = Logger(file_path)

def console_print(message, bold=False):
    logger.print(message, bold)

def console_print_bold(message):
    logger.print_bold(message)

def save_log():
    logger.save_log()


def save_model_architecture(model, file_path):
    # Redirect stdout to a string buffer
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    # Print the model, which will now be captured in the buffer
    print(model)

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output as a string
    model_architecture = buffer.getvalue()

    # Write the captured output to a file
    with open(file_path, 'w') as f:
        f.write(model_architecture)

    print(f"Model architecture saved to {file_path}")