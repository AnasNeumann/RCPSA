# =================================
# =*= GLOBAL CONFIGURATION FILE =*=
# =================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

INTERACTIVE: bool        = False

INFINITY: int            = 100000

TASK_ID_DIM: int         = 4
RESOURCE_ID_DIM: int     = 2
EMBEDDING_DIMENSION: int = 28 # 16, 12
ATTENTION_HEADS: int     = 4
GNN_STACK_SIZE: int      = 2
DROPOUT_RATE: float      = 0.15

MEMORY_CAPACITY: int     = 40_000 # number of transitions in the replay memory
NB_EPISODES: int         = 6_000  # number of training/solving episodes
BATCH_SIZE: int          = 256    # size of each batch sampled from the replay memory
TOP_K: int               = 4      # number of top-Q actions to consider in the Boltzmann exploration
TEMPERATURE: float       = 0.45   # temperature parameter for the Boltzmann exploration
LR: float                = 1e-4   # learning rate of AdamW (Init value: 1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6)
MIN_LR: float            = 1e-5   # min learning rate
PATIENCE: int            = 500    # number of episodes without loss improvement before reducing the learning rate
EPS_START: float         = 0.99   # starting value of epsilon
EPS_END: float           = 0.005  # final value of epsilon
EPS_DECAY: int           = 350    # controls the rate of exponential decay of epsilon, higher means a slower decay (≈35%)

GREEDY_RATE: float       = 0.3    # rate of using greedy action selection instead of stochastic action selection

GAMMA: float             = 1.0    # discount factor
TAU: float               = 0.003  # update rate of the target network

W_LB: float              = 0.5   
W_UB: float              = 0.4
W_FINAL: float           = 0.1

ELITS: int               = 250    # maximum number of elits solutions for the X operator

O: str = "operation"
R: str = "resource"
P: str = "precedence"
D: str = "requires"
S: str = "successors"