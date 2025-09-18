# =================================
# =*= GLOBAL CONFIGURATION FILE =*=
# =================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

INTERACTIVE: bool        = False

EMBEDDING_DIMENSION: int = 16 # 12
ATTENTION_HEADS: int     = 4
GNN_STACK_SIZE: int      = 2
DROPOUT_RATE: float      = 0.15

MEMORY_CAPACITY: int     = 30_000 # number of transitions in the replay memory
NB_EPISODES: int         = 1_000  # number of training/solving episodes
BATCH_SIZE: int          = 128    # size of each batch sampled from the replay memory
TOP_K: int               = 5      # number of top-Q actions to consider in the Boltzmann exploration
TEMPERATURE: float       = 0.95   # temperature parameter for the Boltzmann exploration
LR: float                = 1e-4   # learning rate of AdamW (Init value: 1e-4, 5e-5, 2.5e-5, 1.25e-5, 6.25e-6)
EPS_START: float         = 0.99   # starting value of epsilon
EPS_END: float           = 0.005  # final value of epsilon
EPS_DECAY: int           = 250    # controls the rate of exponential decay of epsilon, higher means a slower decay (≈35%)

INT_AND_DIV_RATE: float  = 0.3    # rate of using intensification based on lower_bound and diversification based on nb visits
GREEDY_RATE: float       = 0.3    # rate of using greedy action selection instead of stochastic action selection

GAMMA: float             = 1.0    # discount factor
TAU: float               = 0.003  # update rate of the target network

W_FINAL:float            = 1.0    # weight of the final makespan in the reward function
W_NON_FINAL:float        = 0.2    # weight of the non-final makespan in the reward function
W_DURATION:float         = 0.1    # weight of the duration increase in the reward function

MAX_ITTRS: int           = 3      # maximum number of iterations in the local search  
ELITS:int                = 200    # maximum number of elits solutions for the X operator

O: str = "operation"
R: str = "resource"
P: str = "precedence"
D: str = "requires"
S: str = "successors"