# =================================
# =*= GLOBAL CONFIGURATION FILE =*=
# =================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

INTERACTIVE: bool        = True

EMBEDDING_DIMENSION: int = 12
ATTENTION_HEADS: int     = 4
GNN_STACK_SIZE: int      = 2
DROPOUT_RATE: float      = 0.1

MEMORY_CAPACITY: int     = 65000 # number of transitions in the replay memory
NB_EPISODES: int         = 6000  # number of training/solving episodes
BATCH_SIZE: int          = 256   # size of each batch sampled from the replay memory

LR: float                = 5e-4  # learning rate of AdamW

GAMMA: float             = 1.0   # discount factor
TAU: float               = 0.003 # update rate of the target network

W_FINAL_MAKESPAN:float   = 0.65
W_DELTA_MAKESPAN:float   = 0.35

O: str = "operation"
R: str = "resource"
P: str = "precedence"
D: str = "requires"
S: str = "successors"