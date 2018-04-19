'''
Morgan Bryant, April 2018
Rebuilding, using my old code, a new experiment launcher system that works
smoother from the very start. See README and commit messages for more info.
''' 


#sys.path.append('./baselines/common/')


import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys, time, datetime, os
from PathfinderEnv import PathEnv, pr_st, print_state
from expenv import ExpAPI

from Config import Config

#def model(inpt, num_actions, scope, reuse=False):
#    print('~!~!~!~!~!~!~!~ Model made here. inpt,num_actions:', inpt, inpt.shape, num_actions)
#    """This model takes as input an observation and returns values of all actions."""
#    with tf.variable_scope(scope, reuse=reuse):
#        out = inpt
#        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
#        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
#        return out


def run_exp_save(envir, centrism, seed, dest, expl, curr, repr_seed, \
                    done_score, trial_counter, actn, mna, fout):

    s = 'params::: envir, centrism, seed, dest, expl, curr, repr_seed, '+\
        'done_score, trial_counter, actn_mode, max_num_actions'
    
    if not fout==None: fout.write(s+'\n')
    if Config.SAVE_LOGS:print(s); 
    for field in [envir, centrism, seed, dest, expl, curr, repr_seed, \
                    done_score, trial_counter, actn, mna]:
        if not fout==None: fout.write(str(field)+'\n')
        if Config.SAVE_LOGS:print('ESSENTIAL PARAMETER:', field)

    if not fout==None: 
        fout.write('----arguments done----\nepisode // test reward\n\n')

    with U_custom.make_session(8) as sess:
        # Create the environment
        env = PathEnv(ExpAPI(envir, centrism, card_or_rot=actn), envir)
        print('>&>', env.observation_space, type(env.observation_space))
        #help(env.observation_space)
        print('>%>', env.observation_space.shape)
        # Create all the functions necessary to train the model


        ''' This commented model works indeed.  
        model = cnn_to_mlp(convs=[env.observation_space.shape], hiddens=[64],\
                    dueling=True, layer_norm=False)
        '''
        sess.run(tf.global_variables_initializer())
        if centrism in ['allocentric', 'egocentric']:
            model = original_pathfinder_model(seed=seed, config=Config) # network, that is ('model' is bad name)
            act, train, update_target, debug = deepq.build_train(
                make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-6),
                gamma=0.9,
                reuse=tf.AUTO_REUSE
            )
        elif centrism == 'choose-mixed':
            model = decider_two_model(seed=seed, config=Config) # network, that is ('model' is bad name)
            print(env.observation_space.shape)
            env.observation_space.shape = tuple(list(env.observation_space.shape)+[2])
            act, train, update_target, debug = build_train_paired(
                make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-6),
                gamma=0.9,
                reuse=tf.AUTO_REUSE
            )

        # Create the replay buffer
        replay_buffer = ReplayBuffer(1000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        #exploration = LinearSchedule(schedule_timesteps=1000, initial_p=1.0, final_p=0.02)

        if len(expl)>4 and expl[:4]=='FLAT':
            exploration = ConstantSchedule(float(expl.split(':')[1]))
        elif len(expl)>6 and expl[:6]=='LINEAR-1':
            startp,endp,nsteps = (float(f) for f in expl.split(':')[1:])
            exploration = LinearSchedule(schedule_timesteps=nsteps, initial_p=startp, final_p=endp)


        # Initialize the parameters and copy them to the target network.
        U.initialize()
        # update_target() -> repeated fresh trials *don't* want to set .....
        episode_rewards = [0]
        testing_results = [0]
        num_actions_taken = [0]
        obs = env.reset(t=0, curr=curr, test_train='train')
        if Config.DEBUG:
            print(type(obs))
            print_state(obs)
        mode_te_tr = 'train'
        test_t=train_t=0
        is_solved=is_episode_completed=False


        for t in itertools.count():
            # Take action and update exploration to the newest value
            if mode_te_tr=='train': train_t += 1
            elif mode_te_tr=='test': test_t += 1

            if Config.DEBUG:
                print(mode_te_tr, type(obs))
                print('STEP',t, train_t,mode_te_tr, is_episode_completed, is_solved, np.mean(episode_rewards[-101:-1]), len(episode_rewards), len(testing_results), Config.TEST_FREQ, Config.TEST_SAMPLE_SIZE  )#, episode_rewards, testing_results)
                print_state(obs)
                time.sleep(1)

            if mode_te_tr=='train':
                action = act(obs[None], update_eps=exploration.value(train_t))[0]
            elif mode_te_tr=='test':
                action = act(obs[None], update_eps=0)[0]
            # ^ set the epsilon according to the step count and whether 
            # it's training/test mode
            new_obs, rew, is_episode_completed, _ = env.step(action)

            num_actions_taken[-1] += 1
            if mna>0 and num_actions_taken[-1]>=mna:
                is_episode_completed = True # Maximum number of actions is reached.
            #rew=rew-0.01 # MORGAN HACK to encourage fast action selection
            # Store transition in the replay buffer.
            
            if mode_te_tr=='train':
                replay_buffer.add(obs, action, rew, new_obs, float(is_episode_completed))
            obs = new_obs
            if is_episode_completed:
                if Config.DEBUG: print('resetting')
                obs = env.reset(t=len(episode_rewards), curr=curr, test_train=mode_te_tr)
                if mode_te_tr=='test' and Config.DEBUG: print(test_t, rew)

            if mode_te_tr=='train':     episode_rewards[-1] += rew
            elif mode_te_tr=='test':    testing_results[-1] += rew
            else: raise Exception()

#                is_solved = train_t > 100 and \
#                                np.mean(episode_rewards[-101:-1]) >= \
#                                done_score * (1-exploration.value(train_t)) 
            is_solved = train_t > 100 and done_score <= np.mean(testing_results[-101:-1]) \
                            and len(testing_results)>100

                            #and exploration.value(train_t)<0.8 
            if is_episode_completed and mode_te_tr=='train': episode_rewards.append(0)
            if is_episode_completed and mode_te_tr=='test':  testing_results.append(0)
            if is_episode_completed and mode_te_tr=='train': num_actions_taken.append(0)


            if is_solved or len(episode_rewards)>Config.MAX_NUM_EPISODES:
                if is_episode_completed: 
                # Show off the result
                    print("MODEL COMPLETED: action",action)
                    return 
                if (is_episode_completed or train_t>200) and \
                            Config.VISUALIZATION:
                    print ('\n\nreward:',rew,'\nnew trial\n')
                    env.render() 
                    time.sleep(1)
                #if is_episode_completed: print (obs,'\n\nnew trial\n')
                # Minimize the error in Bellman's equation on a batch 
                # sampled from replay buffer.

            if mode_te_tr=='train':
                if train_t > 60: #1000:
                    obses_t, actions, rewards, obses_tp1, dones = \
                            replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones,\
                            np.ones_like(rewards))
                # Update target network periodically.
                if 0== train_t % 60: # 1000:
                    update_target()
                    
#                if is_episode_completed and mode_te_tr=='test' and \
#                            len(testing_results) % Config.TEST_SAMPLE_SIZE == 0:
            if mode_te_tr=='test' and is_episode_completed and \
                        len(testing_results) % Config.TEST_FREQ == 0:
                if not fout==None:
                    fout.write(str(len(testing_results))+'\t'+str(round(np.mean(\
                                        testing_results[-101:-1]), 3))+'\n')
                logger.record_tabular("steps", train_t)
                logger.record_tabular("episodes", len(testing_results))
                logger.record_tabular("mean test reward", round(np.mean(\
                                        testing_results[-101:-1]), 3))
                logger.record_tabular("mean train reward", round(np.mean(\
                                        episode_rewards[-101:-1]), 3))
                logger.record_tabular("average number of actions taken", round(np.mean(\
                                        num_actions_taken[-101:-1]), 3))
                logger.record_tabular("% time spent exploring", int(100 * \
                                        exploration.value(train_t)))
                logger.dump_tabular()
            
            if mode_te_tr=='train' and (len(episode_rewards)+1) % Config.TEST_FREQ == 0: 
                    mode_te_tr = 'test'
            elif mode_te_tr=='test' and (len(testing_results)+1) % Config.TEST_SAMPLE_SIZE == 0: 
                    mode_te_tr = 'train'
            if mode_te_tr=='test' and Config.DEBUG:
                print( '\t',len(testing_results))


def _run_expt(envir, centrism, seed, dest, expl, curr, repr_seed, done_score, \
                    trial_counter, actn, mna):
    assert(envir in ['r-u', 'r-u-ru'])
    assert(centrism in ['allocentric', 'egocentric','choose-mixed'])
    assert(actn in ['card', 'rot']) # don't compare these across steps
#    results_outputfile_name = './result_logs/res-'+\
#                        str(time.gmtime().tm_mon)+'-'+\
#                        str(time.gmtime().tm_mday)+'-'+\
#                        str(time.gmtime().tm_hour)+'-'+\
#                        str(time.gmtime().tm_min)+'-'+\
#                        str(envir)+'-'+str(centrism)+'.txt'

    if Config.SAVE_LOGS == True:
        results_outputfile_name = os.path.join(dest,\
                    'res-'+ str(datetime.datetime\
                    .now().strftime("%H_%M_%S"))+'--'+\
                    str(envir)+'--'+str(centrism)+'.txt')

        with open(results_outputfile_name, 'w') as fout:
            run_exp_save( envir, centrism, seed, dest, expl, curr, repr_seed,\
                          done_score, trial_counter, actn, mna, fout=fout)
    else:
        run_exp_save( envir, centrism, seed, dest, expl, curr,\
                  repr_seed, done_score, trial_counter, actn, mna, fout=None)



def run_single_expt(config_inp=None):
    # This function is isolated out to encourage the following workflow:
    # for each experiment, set Config file, and call this 'monolithically'
    _run_expt(  envir = config_inp.GAME_NAME,   \
                curr = config_inp.CURRICULUM,   \
                dest = config_inp.OUTPUT_DIRECTORY,  \
                actn = config_inp.ACTION_MODE,  \
                expl = config_inp.EXPLORATION_SCHEDULE,  \
                centrism = config_inp.CENTRISM,  \
                seed = config_inp.SEED, \
                repr_seed = config_inp.SET_RANDOM_SEED, \
                done_score = config_inp.DONE_SCORE, \
                mna = config_inp.MAX_NUM_ACTIONS, \
                trial_counter = config_inp.TRIAL_COUNTER,
                ) 

if __name__ == '__main__':
    for s in [0,1]:
        Config.SEED = s
        run_single_expt(Config)
import custom_deepq
from Config import Config
from datetime import datetime
import os, shutil

# Setup save directories and store settings, etc
if Config.SAVE_LOGS:
    output_directory = './result_logs/result_logs_'+datetime.now()\
            .strftime('%m-%d-%y_%H%M%S')+'/'
    os.makedirs(output_directory)
    shutil.copyfile("./launch_experiment.py", \
            output_directory+'launch_experiment--generator.py')
    shutil.copyfile("./Config.py", \
            output_directory+'Config--generator.py')

    Config.OUTPUT_DIRECTORY = output_directory
else:
    print ("NOTICE: SAVE_LOGS is off and no results are being saved.")
    Config.OUTPUT_DIRECTORY = None

'''
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
trial_counter = 0
##################################################
# Only change code below this line for experiments.  
'''
for envir in ['r-u-ru']:
 for cent in ['egocentric','allocentric']:
  for curr in ['LINEAR-1:1.0:0.2:500', 'STEP-1:1.0:0.2:250', 'FLAT:0.2']:
   for seed in range(40):
    for expl in ['FLAT:0.5']:
     for actn in ['card','rot']:
      for mna in [-1,2]:
'''
for envir in ['r-u-ru']:
 for cent in ['choose-mixed']:
  for curr in ['FLAT:0.5']:
    for expl in ['FLAT:0.5']:
     for actn in ['card']:
      for mna in [2]:
# Only change code above this line for experiments.  
##################################################
    #for expl in ['flat:0.5','flat:0.8','linear-1:1.0:0.2:250']:

                Config.GAME_NAME = envir
                Config.MAX_NUM_ACTIONS = mna
                Config.CURRICULUM = curr
                Config.ACTION_MODE = actn
                Config.EXPLORATION_SCHEDULE = expl
                Config.CENTRISM = cent
                Config.SEED = seed
                Config.TRIAL_COUNTER = trial_counter
                trial_counter += 1
                custom_deepq.run_single_expt(Config)

