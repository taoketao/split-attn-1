# Configuration file for experiment options, hyperparameters, etc
# This should hold _default_ values that are explicitly not to 
# change for certain experiments.

# See ./launch_experiment.py for a more robust setting.
# These are default values, set here also because have little to
# do with the reference frame experiments.

# Originally, ITEM INITIALIZER was the only part initialized in 
# a function, but for logging purposes, it works best to simply
# make all printable fields self-fields (a python language detail).

class Config: # class'd for ease of access
  def __init__(self):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    '''''   SYSTEM OPTIONS 
            purely programmatic and interface options
    '''''
 
    self.VISUALIZATION = False # for interactive or for bulk-processing modes
    self.PLOTTING_SMOOTHING = 10

    self.SAVE_LOGS = False
    self.ALT_SAVE_DIR = None
    self.DEBUG = False # what, specifically???

    self.MACHINE = 'unspecif' # or laptop or desktop
    # Desktop options
    self._DEVICE = '/device:GPU:0' # desktop
    self._NCORESMAX = 5
    # Laptop options
    self._DEVICE = '/cpu:0' # laptop
    self._NCORESMAX = 2
    # Unspecif options
    self._DEVICE = None 
    self._NCORESMAX = 1

    self._LOG_DEVICE_PLACEMENT = False # debug..?
    self._GPU_ALLOW_GROWTH = True
    self._GPU_FRACTION = 0.8
    self._MAXNTHREADS = 8
    
    # @ tensorflow:
    self.SCOPE = None # True default value. Use string [stub: keywords]
    self.REUSE = 'default'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    '''''   EXPERIMENT OPTIONS 
            identify and build the unique experiment
    '''''

    self.GAME_NAME = 'gould-card-1'
    # GAME_NAME: what ID'd environment is being experienced.
    # Curriculum orderings are implicit. options: 'r-u' 'r-u-ru'

    self.CENTRISM = 'choose-mixed' #'forked' 
    # ^ options: 'egocentric'  'allocentric'  'forked', 'choose-mixed'
    self.ACTION_MODE = 'card' # vs. rotational

    self.CURRICULUM_NAME = 'FLAT'
    # options: 'FLAT' (0 param), 'STEP' (1 param), 'GRADUAL' (2 param),
    # 'FLAT:0.5', ...    See optimization options information block
    self.CURRICULUM_PARAM_1 = None
    self.CURRICULUM_PARAM_2 = None

    self.DONE_MODE = 'epochs' # cp 'score'
    #DONE_SCORE = 0.96 # score needed to be considered 'learned'
    self.DONE_EPOCHS = 2000 # instead, number of epochs to train for

    self.DATA_MODE = 'shuffled' # would need a good reason to do otherwise
    self.TEST_FREQ = 10 # orig: 10
    self.TEST_SAMPLE_SIZE = 10

    #SET_RANDOM_SEED= True # DO not change without changing custom_models
    self.SET_SEED_MANUALLY = False
    self.SEED = None

    self.NETWORK = None #  identify the network topology via keyword.
    self.TRIAL_COUNTER = 'stub' # iterate this for multitrials, with a config
#                   varying tool
    # network manual specification:
    self.NETWORK_STRUCTURE = 'jjb-wide' # jordan jacobs barto: 1 layer attends 2 others
    # hypothesized: jjb-wide is attn gets both allo and ego; jjb-mix is it gets sum
    self.NET_ATTN_LAYERS = [60]
    self.NET_ALLO_LAYERS = [61,62]
    self.NET_EGO_LAYERS = [63,59]

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
    self.MAX_NUM_ACTIONS = 4
    #MAX_NUM_EPISODES = 2e3  <- see DONE_EPOCHS etc
    self.LEARNING_RATE = 1e-4 # or, indicate a schedule
    self.EPSILON_EXPLORATION = 8e-1 # or, indicate a schedule; such as 'FLAT:0.5',
    # 'FLAT:0.8', 'linear-1:1.0:0.2:250', ...
    self.OPTIMIZER = ['adam', 1e-6]
    self.LOSS_FUNCTION = 'huber3e-5'
    self.EPSILON = 1.0
    self.REWARD = 1.0
    self.NO_REWARD = 0.0
    self.INVALID_REWARD = 0.0
    self.GAMMA = 0.9
    

    ''' 
    interim default parameters, before a network-config-populator
    is ready. '''

    self.LAYER_NORM = True # layerwise norm: see tf.contrib.layers.layer_norm
    self.DROPOUT = 0.5
    self.LAYER_ACTIVATIONS = None # implementation stub! no activations.
    # depends on q: activate before or after [dropout,norm,etc]?
    self.FINAL_ACTIVATION = 'sigmoid'#'tanh'
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # ITEM INITIALIZER for a few dependent parameters, etc.
    self.CURRICULUM = self.CURRICULUM_NAME
    if self.CURRICULUM_PARAM_1:
        self.CURRICULUM += ':'+self.CURRICULUM_PARAM_1
    if self.CURRICULUM_PARAM_2:
        self.CURRICULUM += ':'+self.CURRICULUM_PARAM_2
    if not self.SCOPE: self.SCOPE = 'default_scope'

