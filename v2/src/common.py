import time
from datetime import datetime

import torch
from torch import Tensor

# =================================================
# =*= COMMON TOOLS AND FUNCTIONS OF THE PROJECT =*=
# =================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def print_to_endstream(text: str, path: str = None) -> None:
        f = open(path, "a")
        f.write(text)
        f.close()

def display_final_computing_time(start_time: float, instance: str, path: str, use_end_stream: bool = True):
    now            = datetime.now()
    dt_string      = now.strftime("%Y_%m_%d_%H_%M_%S")
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time} seconds")
    if use_end_stream:
        print_to_endstream(f"\nExecution time: {execution_time} seconds", f"{path}/data/logs/{instance}.{dt_string}.log")

def to_bool(v: str) -> bool:
    return v.lower() in ['true', 't', 'yes', '1']
