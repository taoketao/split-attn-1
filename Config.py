# Configuration file for experiment options, hyperparameters, etc
# This should hold _default_ values that are explicitly not to 
# change for certain experiments.

# See ./launch_experiment.py for a more robust setting.
# These are default values, set here also because have little to
# do with the reference frame experiments.

class Config: # class'd for ease of access

    #####   SYSTEM OPTIONS 
    #####    - purely programmatic and interface options

    VISUALIZATION = False # for interactive or for bulk-processing modes
    PLOTTING_SMOOTHING = 10

    SAVE_LOGS = False
    DEBUG=False

    _DEVICE = '/device:GPU:0' # desktop
    _DEVICE = '/cpu:0' # laptop
    

    #####   EXPERIMENT OPTIONS 
    #####    - identify and build the unique experiment
    GAME_NAME = 'gould-card-1'
    # GAME_NAME: what ID'd environment is being experienced.
    # Curriculum orderings are implicit. options: 'r-u' 'r-u-ru'


    CENTRISM = 'forked' # options: 'egocentric'  'allocentric'  'forked'
    ACTION_MODE = 'card' # vs. rotational

    CURRICULUM_NAME = 'FLAT'
    # options: 'FLAT' (0 param), 'STEP' (1 param), 'GRADUAL' (2 param)
    CURRICULUM_PARAM_1 = -1
    CURRICULUM_PARAM_2 = -1

    DONE_MODE = 'epochs' # cp 'score'
    #DONE_SCORE = 0.96 # score needed to be considered 'learned'
    DONE_EPOCHS = 2000 # instead, number of epochs to train for

    DATA_MODE = 'shuffled' # would need a good reason to do otherwise
    TEST_FREQ = 10 # orig: 10
    TEST_SAMPLE_SIZE = 10

    SET_RANDOM_SEED= True # DO not change without changing custom_models

    NETWORK = 'stub: identify the network topology.'

    #####   OPTIMIZATION OPTIONS 
    #####    - hyperparameters to improve an experiment's success
    
    MAX_NUM_ACTIONS = 4
    #MAX_NUM_EPISODES = 2e3  <- see DONE_EPOCHS etc
    LEARNING_RATE = 1e-4 # or, indicate a schedule
    EPSILON_EXPLORATION = 8e-1 # or, indicate a schedule
    OPTIMIZER = ['adam', 1e-6]
    LOSS_FUNCTION = 'huber3e-5'
    EPSILON = 1.0
    REWARD = 1.0
    NO_REWARD = 0.0
    INVALID_REWARD = 0.0
    GAMMA = 0.9
    
