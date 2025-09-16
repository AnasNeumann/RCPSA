# =================================
# =*= GLOBAL CONFIGURATION FILE =*=
# =================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

INTERACTIVE: bool        = True

EMBEDDING_DIMENSION: int = 36 # 12
ATTENTION_HEADS: int     = 4
GNN_STACK_SIZE: int      = 3  # 2
DROPOUT_RATE: float      = 0.15

MEMORY_CAPACITY: int     = 30_000 # number of transitions in the replay memory
NB_EPISODES: int         = 1_000  # number of training/solving episodes
BATCH_SIZE: int          = 256    # size of each batch sampled from the replay memory
TOP_K: int               = 5      # number of top-Q actions to consider in the Boltzmann exploration
TEMPERATURE: float       = 0.95   # temperature parameter for the Boltzmann exploration
LR: float                = 1e-4   # learning rate of AdamW
EPS_START: float         = 0.99   # starting value of epsilon
EPS_END: float           = 0.005  # final value of epsilon
EPS_DECAY: int           = 250    # controls the rate of exponential decay of epsilon, higher means a slower decay (≈35%)

CANDIDATES: int          = 2      # number of candidate tasks to consider for the model-based planning
INTENSIFY_RATE: float    = 0.3    # rate of using model-based planning (exploration) instead of the policy network (exploitation)
GREEDY_RATE: float       = 0.2    # rate of using greedy action selection instead of stochastic action selection
INTENSIFY_INC: float     = 1.90   # the more a Q value is near the end of episode, the more it should be trusted!

GAMMA: float             = 1.0    # discount factor
TAU: float               = 0.003  # update rate of the target network

W_FINAL:float            = 1.0    # weight of the final makespan in the reward function
W_NON_FINAL:float        = 0.05   # weight of the non-final makespan in the reward function

O: str = "operation"
R: str = "resource"
P: str = "precedence"
D: str = "requires"
S: str = "successors"