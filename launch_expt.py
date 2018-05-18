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



class network_util(object):
    ''' network_util: a simple (wrapper-like) class that facilitates 
        varied interfacing with the constituent network.
        
    '''
    def __init__(self, conf, model, inp_shape, sess, inp_vars):
        self.c = conf
        self.model=model
        self.inp_shape=inp_shape
        self.sess=sess
        self.inp_vars = inp_vars

        self.nactions = {True:3,False:4}[self.c.ELIM_USELESS_ACTIONS]

        ''' ------------ make variables ------------ '''
        self.targ_var = tf.placeholder(tf.float32, [4])
        self.pred_var = self.model # (just a copy)

        self.loss_fn_updates = self._getLossFn(self.pred_var, self.targ_var)
        self.loss_fn_fixed = self._getLossFn(self.pred_var, self.targ_var)
        assert(type(self.c.LEARNING_RATE) == float)
        if self.c.OPTIMIZER[0]=='adam':
            self.optimizer = tf.train.AdamOptimizer(self.c.LEARNING_RATE,\
                             epsilon = self.c.OPTIMIZER[1])
        else: raise Exception("Optimizer support not yet implemented:",\
                             self.c.OPTIMIZER)
        self.updates = self.optimizer.minimize(self.loss_fn_updates)

    def getQVals(self, state, agency_mode='NEURAL'):
        if agency_mode=='NEURAL':
            return self._forward_pass(state)
        elif agency_mode=='RANDOM':
            return {3: [3**-1, 3**-1, 0, 3**-1], \
                    4: [0.25, 0.25, 0.25, 0.25]} [self.nactions]

    def _getLossFn(self, prd, trg):
        lf = self.c.LOSS_FUNCTION
        if lf == 'square' or lf==None:
            return tf.reduce_sum(tf.square( pred - targ ))
        elif 'huber' in lf:
            if len(lf)>5: max_grad = float(lf[5:])
            else: max_grad = c.DEFAULT_HUBER_SATURATION

            err = tf.reduce_sum(tf.abs( prd - trg ))
            mg = tf.constant(max_grad, name='max_grad')
            lin = mg*(err-.5*mg)
            quad = .5*err*err
            return tf.where(err < mg, quad, lin)
        else: raise Exception("Loss function not acknowledged: "+str(lf))

    def _forward_pass(self, state):
        inp = tf.stack(obs['state'] for obs in state)
        #Q_sa = self.sess.run(self.input

#        for k,v in state[0].items():
#            try: print (k, v.shape)
#            except: print('DEFAULT', k,v)
#
##x.exp_env.state_input_shape()
#        print(self.model)
#        print(self.model.shape)
#        print(len(state))
#        print(state)
#        help(state)
        print(self.inp_vars.shape)
        print(inp.shape)
        print(self.model.shape)
        Q_sa = self.sess.run(self.model, \
                feed_dict = {self.inp_vars : inp});
        return Q_sa








def run_trial( c, net, envs, mode_te_tr):
    """ 
    Parameters
    ----------
    takes config instance, network container, environments, epoch number, 
    test/train flag.
    >> is epoch number needed? isnt that the point of this function?

    Does
    ----
    implements reinforce algorithm as vanilla DQN (for now).
    currently, *not* batch, for sake of time to implement.

    Returns
    -------
    a well-organized struct (dict?) of results of this trial 

    """        
    _a, _e, _a_e = 0,1,0 # local flags

    actions_taken = []
    i = np.random.randint(envs[_a_e].n_train_states)
    start_state = [envs[key].exp_env.train_states[i] for key in [_a,_e]]
    obs_a = [start_state[_a]]; obs_e = [start_state[_e]] # observations

    assert(mode_te_tr=='train', 'not impl yet stub in progress')
    assert(start_state[_a]['startpos'] == start_state[_e]['startpos'])
    assert(start_state[_a]['goalpos']  == start_state[_e]['goalpos'])

    #for XX in [env_allo.exp_env.train_states, env_allo.exp_env.test_states]:

    for itr in itertools.count(): # increment actions
        if type(c.MAX_NUM_ACTIONS)==int and itr >= c.MAX_NUM_ACTIONS:
            break
        ''' ------------ boiler plate: start loop ------------ '''
        Q0_s = net.getQVals(start_state)
        

        ''' ------------ boiler plate: end loop ------------ '''









def new_run_exp_save( c, fout=sys.stdout, replay_buf=False ):
    """ new_run_exp_save: a new function that more directly
         trains (and tests; todo) gould Gold experiments.

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
    #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  
    ''' ------------ initial logging & printing ------------ '''

    print("A new experiment is invoked. Info: first are parameter settings, ")

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

    ''' ------------ (tensorflow) config ------------ '''
    cproto = tf.ConfigProto()
    cproto.log_device_placement = c._LOG_DEVICE_PLACEMENT
    cproto.inter_op_parallelism_threads = c._MAXNTHREADS
    cproto.intra_op_parallelism_threads = c._MAXNTHREADS
    if c.MACHINE=='desktop': 
        cproto.gpu_options.allow_growth = c._GPU_ALLOW_GROWTH
        cproto.gpu_options.per_process_memory_fraction = c._GPU_FRACTION

    with tf.Session(config = cproto ) as sess:
        var_name_id = 1234

        ''' ------------ set environment ------------ '''

        env_allo, env_ego = envs = [PathEnv(c, ExpAPI(c.GAME_NAME, \
                        ctr+'centric', card_or_rot = c.ACTION_MODE)) \
                        for ctr in ('allo','ego')]
 
        ''' ------------ Build JJB model ------------ '''

        _ash, _esh = [x.exp_env.state_input_shape() for x in envs]
        if not c.NETWORK_STRUCTURE=='jjb-wide': 
            raise NotImplemented('jjb-wide is only implemented model')

        model_ego, inp_ego_var = mlp(c=c, inpt_shape=_esh, itr=var_name_id, \
                         vers = 'ego', hiddens = c.NET_EGO_LAYERS)

        model_allo, inp_allo_var = mlp(c=c, inpt_shape=_ash, itr=var_name_id, \
                         vers = 'allo', hiddens = c.NET_ALLO_LAYERS)

        assert(_ash[0]==_esh[0] and _ash[1]==_esh[1])
        attn_shape = tuple([_ash[0], _ash[1], _ash[2]+_esh[2]])
        model_attn, attn_var = mlp(c=c, inpt_shape=attn_shape, osize=2, \
                         itr=var_name_id, debug=True,\
                         vers = 'attn', hiddens = c.NET_ATTN_LAYERS)

#       As implemented, attn is additive not proportional: see config settings.

        assert(c.ATTENTION_MODE_TRAIN == 'smooth-zero-sum')
        model = tf.add(\
                tf.multiply(model_attn[0], model_allo), \
                tf.multiply(model_attn[1], model_ego))
        net = network_util(c, model, attn_shape, sess, \
                    tf.stack([inp_ego_var, inp_allo_var], name='Input'))

        sess.run(tf.global_variables_initializer()) # after env init'ed

        if c.DEBUG:
            print(type(obs_e))
            print_state(obs_e)
            print_state(obs_a)
            for XX in [ env_ego.exp_env.test_states, \
                        env_ego.exp_env.train_states]:
              for s in XX:
                print_state(s)
                print('')
              print('---------')
            print('^ego, v allo')
            for XX in [env_allo.exp_env.train_states, env_allo.exp_env.test_states]:
              for s in XX:
                print('')
                print_state(s)

        mode_te_tr = 'train' # for the moment
        test_t=train_t=0

        for t in itertools.count(): # increment quasi-epochs
            if c.DONE_MODE == 'epochs' and t > c.DONE_EPOCHS:
                break
            # Future: here is a spot to put weight saving @ tensorflow.
            # Future: here is a spot to adjust learning curriculum schedule.    
            
            # Take action and update exploration to the newest value
            if mode_te_tr=='train': train_t += 1
            elif mode_te_tr=='test': test_t += 1

            ''' ------------ Run train ------------ '''

            results = run_trial( c, net, envs, mode_te_tr)
            
            
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

