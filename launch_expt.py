'''
Morgan Bryant, April 2018
Rebuilding, using my old code, a new experiment launcher system that works
smoother from the very start. See README and commit messages for more info.
''' 


#sys.path.append('./baselines/common/')

from __future__ import print_function
from environment import PathEnv, pr_st, print_state, ExpAPI, mlp # convenience
from environment import *
import itertools
import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys, time, os
from datetime import datetime
import os, shutil
from  string import ljust
import textwrap

from Config import Config


#def run_exp_save(envir, centrism, seed, dest, expl, curr, repr_seed, \
#                    done_score, trial_counter, actn, mna, fout):
#
#       ...  ...  ...  ...  ...
#
#        # Create the replay buffer
#        replay_buffer = ReplayBuffer(1000)
#        # Create the schedule for exploration starting from 1 
#        #(every action is random) down to
#        # 0.02 (98% of actions are selected according to values 
#        #predicted by the model).
#        #exploration = LinearSchedule(schedule_timesteps=1000, 
#        #initial_p=1.0, final_p=0.02)
#
#
#
#        for t in itertools.count():
#            # Take action and update exploration to the newest value
#            if mode_te_tr=='train': train_t += 1
#            elif mode_te_tr=='test': test_t += 1
#
#            if Config.DEBUG:
#                print(mode_te_tr, type(obs))
#                print('STEP',t, train_t,mode_te_tr, is_episode_completed, is_solved, np.mean(episode_rewards[-101:-1]), len(episode_rewards), len(testing_results), Config.TEST_FREQ, Config.TEST_SAMPLE_SIZE  )#, episode_rewards, testing_results)
#                print_state(obs)
#                time.sleep(1)
#
#            if mode_te_tr=='train':
#                action = act(obs[None], update_eps=exploration.value(train_t))[0]
#            elif mode_te_tr=='test':
#                action = act(obs[None], update_eps=0)[0]
#            # ^ set the epsilon according to the step count and whether 
#            # it's training/test mode
#            new_obs, rew, is_episode_completed, _ = env.step(action)
#
#            num_actions_taken[-1] += 1
#            if mna>0 and num_actions_taken[-1]>=mna:
#                is_episode_completed = True # Maximum number of actions is reached.
#            #rew=rew-0.01 # MORGAN HACK to encourage fast action selection
#            # Store transition in the replay buffer.
#            
#            if mode_te_tr=='train':
#                replay_buffer.add(obs, action, rew, new_obs, float(is_episode_completed))
#            obs = new_obs
#            if is_episode_completed:
#                if Config.DEBUG: print('resetting')
#                obs = env.reset(t=len(episode_rewards), curr=curr, test_train=mode_te_tr)
#                if mode_te_tr=='test' and Config.DEBUG: print(test_t, rew)
#
#            if mode_te_tr=='train':     episode_rewards[-1] += rew
#            elif mode_te_tr=='test':    testing_results[-1] += rew
#            else: raise Exception()
#
##                is_solved = train_t > 100 and \
##                                np.mean(episode_rewards[-101:-1]) >= \
##                                done_score * (1-exploration.value(train_t)) 
#            is_solved = train_t > 100 and done_score <= np.mean(testing_results[-101:-1]) \
#                            and len(testing_results)>100
#
#                            #and exploration.value(train_t)<0.8 
#            if is_episode_completed and mode_te_tr=='train': episode_rewards.append(0)
#            if is_episode_completed and mode_te_tr=='test':  testing_results.append(0)
#            if is_episode_completed and mode_te_tr=='train': num_actions_taken.append(0)
#
#
#            if is_solved or len(episode_rewards)>Config.MAX_NUM_EPISODES:
#                if is_episode_completed: 
#                # Show off the result
#                    print("MODEL COMPLETED: action",action)
#                    return 
#                if (is_episode_completed or train_t>200) and \
#                            Config.VISUALIZATION:
#                    print ('\n\nreward:',rew,'\nnew trial\n')
#                    env.render() 
#                    time.sleep(1)
#                #if is_episode_completed: print (obs,'\n\nnew trial\n')
#                # Minimize the error in Bellman's equation on a batch 
#                # sampled from replay buffer.
#
#            if mode_te_tr=='train':
#                if train_t > 60: #1000:
#                    obses_t, actions, rewards, obses_tp1, dones = \
#                            replay_buffer.sample(32)
#                    train(obses_t, actions, rewards, obses_tp1, dones,\
#                            np.ones_like(rewards))
#                # Update target network periodically.
#                if 0== train_t % 60: # 1000:
#                    update_target()
#                    
##                if is_episode_completed and mode_te_tr=='test' and \
##                            len(testing_results) % Config.TEST_SAMPLE_SIZE == 0:
#            if mode_te_tr=='test' and is_episode_completed and \
#                        len(testing_results) % Config.TEST_FREQ == 0:
#                if not fout==None:
#                    fout.write(str(len(testing_results))+'\t'+str(round(np.mean(\
#                                        testing_results[-101:-1]), 3))+'\n')
#                logger.record_tabular("steps", train_t)
#                logger.record_tabular("episodes", len(testing_results))
#                logger.record_tabular("mean test reward", round(np.mean(\
#                                        testing_results[-101:-1]), 3))
#                logger.record_tabular("mean train reward", round(np.mean(\
#                                        episode_rewards[-101:-1]), 3))
#                logger.record_tabular("average number of actions taken", round(np.mean(\
#                                        num_actions_taken[-101:-1]), 3))
#                logger.record_tabular("% time spent exploring", int(100 * \
#                                        exploration.value(train_t)))
#                logger.dump_tabular()
#            
#            if mode_te_tr=='train' and (len(episode_rewards)+1) % Config.TEST_FREQ == 0: 
#                    mode_te_tr = 'test'
#            elif mode_te_tr=='test' and (len(testing_results)+1) % Config.TEST_SAMPLE_SIZE == 0: 
#                    mode_te_tr = 'train'
#            if mode_te_tr=='test' and Config.DEBUG:
#                print( '\t',len(testing_results))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# new experiment launch scripts, adapted from ones above.

def new_run_exp_save( c, fout, replay_buf=False ):
    print("A new experiment is invoked. Info: first are parameter settings, ")
    """ new_run_exp_save: a new function that more directly
        implements a two-headed network.

    Parameters
    ----------
    (well-initialized Config quasi-struct 'c'; system-level fout)

    Does
    ----
      ?

    Returns
    -------
    finished built model
    """        

    try: assert(c.CENTRISM) in ['choose-mixed']
    except: raise Exception(c.CENTRISM+"centrism isn't supported! stub")

    # dump parameters for output: 
    _val=30
    s = ' '.join([ljust(s_,_val) for s_ in [k+':'+str(v) for k,v \
                  in sorted(vars(c).items()) if not k[:2]=='__']])
    itr=0
    #print('questionable process:')
    for si in textwrap.fill(s, 3*_val):
        while not len(si)%_val==1: si += ' '
        print(si, end='')
    print('\n\n')
    sys.stdout.flush()

    ''' tensorflow config -- needs spot check for option 'upgrades' '''
    cproto = tf.ConfigProto()
    cproto.log_device_placement = c._LOG_DEVICE_PLACEMENT
    cproto.inter_op_parallelism_threads = c._MAXNTHREADS
    cproto.intra_op_parallelism_threads = c._MAXNTHREADS
    if c.MACHINE=='desktop':
        cproto.gpu_options.allow_growth = c._GPU_ALLOW_GROWTH
        cproto.gpu_options.per_process_memory_fraction = c._GPU_FRACTION
    ''' end tensorflow config '''

    with tf.Session(config = cproto ) as sess:
        var_name_id = 1234
        # ---- set environment ----
        env_allo = PathEnv(c, ExpAPI(c.GAME_NAME, 'allocentric', card_or_rot = \
                                                              c.ACTION_MODE)) 
        env_ego  = PathEnv(c, ExpAPI(c.GAME_NAME, 'egocentric' , card_or_rot = \
                                                              c.ACTION_MODE)) 
#       in every iteration, there are env_allo and env_ego structures.
        if c.NETWORK_STRUCTURE=='jjb-wide': # build three quasi-JJB model components
            ash, esh = [x.exp_env.state_input_shape() for x in [env_allo, env_ego]]

            model_ego  = mlp(c=c, inpt_shape=esh, itr=var_name_id, \
                             vers = 'ego', hiddens = c.NET_EGO_LAYERS)

            model_allo = mlp(c=c, inpt_shape=ash, itr=var_name_id, \
                             vers = 'allo', hiddens = c.NET_ALLO_LAYERS)

            assert(ash[0]==esh[0] and ash[1]==esh[1])
            attn_shape = tuple([ash[0], ash[1], ash[2]+esh[2]])
            model_attn = mlp(c=c, inpt_shape=attn_shape, osize=2, \
                             itr=var_name_id, debug=True,\
                             vers = 'attn', hiddens = c.NET_ATTN_LAYERS)

#       As implemented, attn interpreted as additive not proportional: 428
            model = tf.add(\
                    tf.multiply(model_attn[0], model_allo), \
                    tf.multiply(model_attn[1], model_ego))
        else: raise NotImplemented('jjb-wide is only implemented model')
        
        sess.run(tf.global_variables_initializer()) # after env init'ed

        episode_rewards = [0]
        testing_results = [0]
        num_actions_taken = [0]
        # 4/19/18: due to immense utility of reset-act-etc openai framework,
        # initial thought says make it like that.
        obs_e = env_ego.reset(t=0, curr=c.CURRICULUM, test_train='train')
        obs_a = env_allo.reset(t=0, curr=c.CURRICULUM, test_train='train')


        if c.DEBUG:
            print(type(obs_e))
            print_state(obs_e)
            print_state(obs_a)
        for XX in [env_ego.exp_env.test_states, env_ego.exp_env.train_states]:
          for s in XX:
#            err_delta = 2-s['startpos'][1]
#            print(err_delta, s.keys())
#            s['state'] = np.roll(s['state'], err_delta, axis=1)
            print_state(s)
            print('')
          print('---------')
        print('^ego, v allo')
        for XX in [env_allo.exp_env.train_states, env_allo.exp_env.test_states]:
          for s in XX:
            print('')
            print_state(s)

        mode_te_tr = 'train'
        test_t=train_t=0
        is_solved=is_episode_completed=False

        sys.exit()

        for t in itertools.count():
            if c.DONE_MODE == 'epochs' and t > c.DONE_EPOCHS:
                break
            # Take action and update exploration to the newest value
            if mode_te_tr=='train': train_t += 1
            elif mode_te_tr=='test': test_t += 1
# .... stub left off 428
            if mode_te_tr=='train': 
                action = act
            elif mode_te_tr=='test': test_t += 1

        print('sanity: done')

def new_launch_expt(config):
    c=config()

    # Input handle:
    assert(c.GAME_NAME in ['r-u', 'r-u-ru', 'gould-card-1'])
    if c.CENTRISM=='forked': c.centrism = 'choose-mixed'
    assert(c.CENTRISM in ['allocentric', 'egocentric','choose-mixed'])
    assert(c.ACTION_MODE in ['card', 'rot'])

    # Setup saving:
    if c.SAVE_LOGS:
        output_directory = './result_logs/result_logs_'+datetime.now()\
                .strftime('%m-%d-%y_%H%M%S')+'/'
        if c.ALT_SAVE_DIR: output_directory = c.ALT_SAVE_DIR
        c.OUTPUT_DIRECTORY = output_directory
        os.makedirs(output_directory)
#        shutil.copyfile("./launch_expt.py", \
#                output_directory+'launch_expt--generator.py')
        shutil.copyfile("./Config.py", \
                output_directory+'Config--generator.py')

        results_outputfile_name = os.path.join(output_directory,\
                    'res-'+ str(datetime\
                    .now().strftime("%H_%M_%S"))+'--'+\
                    str(c.GAME_NAME)+'--'+str(c.CENTRISM)+'.txt')

        with open(results_outputfile_name, 'w') as fout:
            new_run_exp_save( c, fout=fout )
    else:
        print ("NOTICE: SAVE_LOGS is off and no results are being saved.")
        c.OUTPUT_DIRECTORY = None
        new_run_exp_save( c, fout=sys.stdout )


if __name__=='__main__':
    # Run single trial
    Config.TRIAL_COUNTER = '-'
    #custom_deepq.run_single_expt(Config)
    new_launch_expt(Config)

