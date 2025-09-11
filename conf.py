# =================================
# =*= GLOBAL CONFIGURATION FILE =*=
# =================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

MEMORY_CAPACITY  = 65000 # number of transitions in the replay memory

W_FINAL_MAKESPAN = 0.65
W_DELTA_MAKESPAN = 0.35

O: str = "operation"
R: str = "resource"
P: str = "precedence"
D: str = "requires"
S: str = "successors"