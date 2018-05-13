# Configuration file for experiment options, hyperparameters, etc
# This should hold _default_ values that are explicitly not to 
# change for certain experiments.

# See ./launch_experiment.py for a more robust setting.
# These are default values, set here also because have little to
# do with the reference frame experiments.

class Config: # class'd for ease of access

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    '''''   SYSTEM OPTIONS 
            purely programmatic and interface options
    '''''
 
    VISUALIZATION = False # for interactive or for bulk-processing modes
    PLOTTING_SMOOTHING = 10

    SAVE_LOGS = False
    ALT_SAVE_DIR = None
    DEBUG = False # what, specifically???

    MACHINE = 'unspecif' # or laptop or desktop
    # Desktop options
    _DEVICE = '/device:GPU:0' # desktop
    _NCORESMAX = 5
    # Laptop options
    _DEVICE = '/cpu:0' # laptop
    _NCORESMAX = 2
    # Unspecif options
    _DEVICE = None 
    _NCORESMAX = 1

    _LOG_DEVICE_PLACEMENT = False # debug..?
    _GPU_ALLOW_GROWTH = True
    _GPU_FRACTION = 0.8
    _MAXNTHREADS = 8
    
    
    # @ tensorflow:
    SCOPE = None # True default value. Use string [stub: keywords]
    REUSE = 'default'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    '''''   EXPERIMENT OPTIONS 
            identify and build the unique experiment
    '''''

    GAME_NAME = 'gould-card-1'
    # GAME_NAME: what ID'd environment is being experienced.
    # Curriculum orderings are implicit. options: 'r-u' 'r-u-ru'

    CENTRISM = 'choose-mixed' #'forked' 
    # ^ options: 'egocentric'  'allocentric'  'forked', 'choose-mixed'
    ACTION_MODE = 'card' # vs. rotational

    CURRICULUM_NAME = 'FLAT'
    # options: 'FLAT' (0 param), 'STEP' (1 param), 'GRADUAL' (2 param),
    # 'FLAT:0.5', ...    See optimization options information block
    CURRICULUM_PARAM_1 = None
    CURRICULUM_PARAM_2 = None

    DONE_MODE = 'epochs' # cp 'score'
    #DONE_SCORE = 0.96 # score needed to be considered 'learned'
    DONE_EPOCHS = 2000 # instead, number of epochs to train for

    DATA_MODE = 'shuffled' # would need a good reason to do otherwise
    TEST_FREQ = 10 # orig: 10
    TEST_SAMPLE_SIZE = 10

    #SET_RANDOM_SEED= True # DO not change without changing custom_models
    SET_SEED_MANUALLY = False
    SEED = None

    NETWORK = None #  identify the network topology via keyword.
    TRIAL_COUNTER = 'stub' # iterate this for multitrials, with a config
#                   varying tool
    # network manual specification:
    NETWORK_STRUCTURE = 'jjb-wide' # jordan jacobs barto: 1 layer attends 2 others
    # hypothesized: jjb-wide is attn gets both allo and ego; jjb-mix is it gets sum
    NET_ATTN_LAYERS = [60]
    NET_ALLO_LAYERS = [61,62]
    NET_EGO_LAYERS = [63,59]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    '''   OPTIMIZATION OPTIONS 
          hyperparameters to improve an experiment's success
    
    
    Info for parameters that are not self-explanatory:
        envir/TASK_ENVIRONMENT is the 'task' made up of 'challenges'
        curr/CURRICULUM is the task_envir-dependent curriculum scheme.
            param for FLAT is the pct of r and u trials; 1-param is ru chance
        expl/EXPLORATION_SCHEDULE is the epsilon exploration chance

        expl and curr formatted as:
            [FLAT]:[float in (0,1) for const. epsilon exploration]
            [LINEAR-1]:[start %]:[end %]:[n steps interpolated to end]
            [STEP-1]:[start %]:[end %]:[epoch]
            Note linear is only, at present, able to interpolate from 0.
    '''
    MAX_NUM_ACTIONS = 4
    #MAX_NUM_EPISODES = 2e3  <- see DONE_EPOCHS etc
    LEARNING_RATE = 1e-4 # or, indicate a schedule
    EPSILON_EXPLORATION = 8e-1 # or, indicate a schedule; such as 'FLAT:0.5',
    # 'FLAT:0.8', 'linear-1:1.0:0.2:250', ...
    OPTIMIZER = ['adam', 1e-6]
    LOSS_FUNCTION = 'huber3e-5'
    EPSILON = 1.0
    REWARD = 1.0
    NO_REWARD = 0.0
    INVALID_REWARD = 0.0
    GAMMA = 0.9
    

    ''' 
    interim default parameters, before a network-config-populator
    is ready. '''

    LAYER_NORM = True # layerwise norm: see tf.contrib.layers.layer_norm
    DROPOUT = 0.5
    LAYER_ACTIVATIONS = None # implementation stub! no activations.
    # depends on q: activate before or after [dropout,norm,etc]?
    FINAL_ACTIVATION = 'sigmoid'#'tanh'
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __init__(self):
        # initializer for a few dependent parameters, etc.
        self.CURRICULUM = self.CURRICULUM_NAME
        if self.CURRICULUM_PARAM_1:
            self.CURRICULUM += ':'+self.CURRICULUM_PARAM_1
        if self.CURRICULUM_PARAM_2:
            self.CURRICULUM += ':'+self.CURRICULUM_PARAM_2
        if not self.SCOPE: self.SCOPE = 'default_scope'

