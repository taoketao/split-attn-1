'''                                                                           |
Morgan Bryant, April 2018                                                     |
This is a composite module of fundamental implemented environments, 
experiment managers, and wrappers for mass loading into new scripted
process experiments.
'''
import sys, os, random, time
import numpy as np
from scipy.sparse import coo_matrix
import tensorflow as tf
import tensorflow.contrib.layers as layers

from Config import Config

if __name__=='__main__': print('Loading custom resources...',time.asctime())

X=0; Y=1;
XDIM=0; YDIM=0

# layer globals:
agentLayer = 0;
goalLayer = 1;
immobileLayer = 2;
mobileLayer = 3;
NUM_LAYERS = 4; # agent, goal, immobile, mobile
LAYER_MAP = {'A':0, 'G':1, 'I':2, 'M':3}
layer_names = {\
        0:"Agent Layer", \
        1:"Goal Layer", \
        2:"Immobile Block Layer",\
        3: "Mobile Block Layer"}
# OLAYERS: ordered layers, where pos corresponds to value
OLAYERS = [agentLayer, goalLayer, immobileLayer, mobileLayer]
AL,GL,IL,ML = [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]
AL[agentLayer]=1;       AL=np.array(AL)
GL[goalLayer]=1;        GL=np.array(GL)
ML[mobileLayer]=1;      ML=np.array(ML)
IL[immobileLayer]=1;    IL=np.array(IL)
not_a_layer = np.array([0,0,0,0])

# Global sentinels:
MOVE_NORTH, MOVE_SOUTH, MOVE_EAST,  MOVE_WEST = 100,101,102,103
MOVE_FWD,   MOVE_BACK                         = 110,111
ROT_R90_C,  ROT_R180_C,  ROT_R270_C           = 120,121,122

# action globals:
N_ACTIONS = 4
NORTH = UDIR = ACTION0 = 0
EAST = RDIR = ACTION1 = 1
SOUTH = DDIR = ACTION2 = 2
WEST = LDIR = ACTION3 = 3
cardinals = [NORTH, EAST, SOUTH, WEST]
inv_cards = [SOUTH, WEST, NORTH, EAST]
ALL = NO_MOVE = -1
ROT0=10; 
ROT90=11; 
ROT180=12; 
ROT270=13;
rots = [ROT0, ROT90, ROT180, ROT270]
invrots = [ROT180, ROT90, ROT0, ROT270]
id_to_rot = { ROT0:0, ROT90:1, ROT180:2, ROT270:3 }
INDICES_TO_CARDINAL_ACTIONS =\
        { 0:MOVE_NORTH, 1:MOVE_SOUTH, 2:MOVE_EAST, 3:MOVE_WEST }

# Codes:
DVECS = {MOVE_NORTH: (0,-1), MOVE_SOUTH: (0,1), MOVE_EAST: (1,0), \
        MOVE_WEST: (-1,0)}
DIRVECS = {(0,-1):'N', (0,1):'S', (1,0):'E', (-1,0):'W'}

# [Helper] Constants
N_EPS_PER_EPOCH = 4 # upper bound on number of initial start states there are

# flags for state components, which I think are mildly redundant, tag 90:
ARR=0; FLIPLR=1; FLIPUD=2; ROT90=3; ROT180=4; XSZ=5; YSZ=6; ALOC=7; GLOC=8;


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# [formerly templates.py]

'''     Template guide: 
            x   =   immobile block
            m   =   mobile block
            .   =   empty space
            a   =   an agent starting place
            !   =   goal
            h   =   hidden immobile block
            r   =   end line
            e   =   end map
            o   =   start options list, after e
            D   =   'dirt': place mobile blocks over all '!' locs
            *   =   'cross': cross each agent location by each goal location
'''

TEMPLATE_GOULD_TRAIN_1 =  '''\
            . x !         r
            I h .         r
            . x .         r
            . . .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_2 =  '''\
            . x !         r
            I . .         r
            . x .         r
            . h .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_3 =  '''\
            . x .         r
            I . .         r
            . x !         r
            . h .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_4 =  '''\
            . x .         r
            I h .         r
            . x !         r
            . . .         e
                        o: *  '''
#  Who has advantage, ego or allo?
# ability:      1,3: same. 1,4: allo. 2,3: same. 2,4: allo.
# optimality:   1,3: allo. 1,4:[allo] 2,3: same. 2,4:[allo]

TEMPLATE_GOULD_TRAIN_5 =  '''\
            . x !         r
            . h .         r
            . x .         r
            I . .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_6 =  '''\
            . x !         r
            . . .         r
            . x .         r
            I h .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_7 =  '''\
            . x .         r
            . . .         r
            . x !         r
            I h .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_8 =  '''\
            . x .         r
            . h .         r
            . x !         r
            I . .         e
                        o: *  '''
#  Who has advantage, ego or allo?
# ability:    1,5: same. 1, ....
# optimality: 1,3: same. 
                        
#################################################

TEMPLATE_GOULD_TRAIN_1 =  '''\
            . . .         r
            . x !         r
            I h .         r
            . x .         r
            . . .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_2 =  '''\
            . h .         r
            . x !         r
            I . .         r
            . x .         r
            . . .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_3 =  '''\
            . h .         r
            . x .         r
            I . .         r
            . x !         r
            . . .         e
                        o: *  '''
TEMPLATE_GOULD_TRAIN_4 =  '''\
            . . .         r
            . x .         r
            I h .         r
            . x !         r
            . . .         e
                        o: *  '''
# allo @ 1+2. allo @ 2+3. allo @ 1+4. allo @ 1+3



TEMPLATE_TSE = ''' x x x x x x x x x x x  r
                   x x x x x a x x x x x  r
                   x x . . . . . . . x x  r
                   x x . ! . . . . ! x x  r
                   x x . . . . ! . . x x  r
                   x a . . . . . . . a x  r
                   x x . . ! . . . . x x  r
                   x x ! . . . . ! . x x  r
                   x x . . . . . . . x x  r
                   x x x x x a x x x x x  r
                   x x x x x x x x x x x  e
        o: D, *  
                   '''

TEMPLATE_R_U =    '''   x x x x x x x  r
                        x m m m m m x  r
                        x m . ! . m x  r
                        x m . a ! m x  r
                        x m . . . m x  r
                        x m m m m m x  r
                        x x x x x x x  e
                o:  *    '''

TEMPLATE_RU =    '''    x x x x x x x  r
                        x m m m m m x  r
                        x m . . ! m x  r
                        x m . a . m x  r
                        x m . . . m x  r
                        x m m m m m x  r
                        x x x x x x x  e
                o:  *    '''
TEMPLATE_R_U_RU =    '''    x x x x x x x  r
                        x m m m m m x  r
                        x m . ! ! m x  r
                        x m . a ! m x  r
                        x m . . . m x  r
                        x m m m m m x  r
                        x x x x x x x  e
                o:  *    '''

TEMPLATE_GOULD_1 =  ''' . . . r
                        . h * r
                        I . . r
                        . h * r
                        . . . e
                o: *  '''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# [formerly environment.py]

'''
Environment class for creating and handling world states.                     |
environment3.py: this version intends to implement ego/allocentric capability.
'''

class state(object):
    '''          ~~ state class ~~
        An object centered about a game state.  Holds data,
        provides methods, and facilitates any [valid] edits desired.
        Roughly a 'RESTful' class: access is based around get, post, del, put.
    '''

    def __init__(self, gridsize, name=None):
        self.gridsz = gridsize
        if name==None: self.name=None
        else: self.name=name[name.rfind('/')+1:]
        self.lexid = None
        self.sparse = False # coo_matrix format?
        self.grid = np.zeros((self.gridsz[XDIM], self.gridsz[YDIM], \
                NUM_LAYERS), dtype='float32')
        self.lexid=None
 
    ''' For storage, sparsification supported.  For operations, desparsify.'''
    def sparsify(self):
        if self.sparse: return
        self.grid = [coo_matrix(self.grid[:,:,i]) for i in range(NUM_LAYERS)]
        self.sparse=True
    def desparse(self):
        if not self.sparse: return
        self.grid = np.array([self.grid[i].toarray() for i in range(NUM_LAYERS)]) 
        self.sparse=False

    def _post(self, loc, whichLayer): # set grid to on
        assert(len(loc)==2)
        #print "Adding", layer_names[whichLayer], "to loc", loc 
        self.desparse()
        self.grid[loc[XDIM], loc[YDIM], whichLayer] = 1 
    def _del(self, loc, whichLayer):
        assert(len(loc)==2)
        #print "Deleting", layer_names[whichLayer], "from loc", loc
        self.desparse()
        self.grid[loc[XDIM], loc[YDIM], whichLayer] = 0 

    ''' Public methods for the AGENT '''
    def get_agent_loc(self): return self.a_loc;
    def post_agent(self, agent_loc):
        self._post(agent_loc, agentLayer)
        self.a_loc = agent_loc
    def put_agent(self, new_loc):
        self._del(self.a_loc, agentLayer)
        self.post_agent(new_loc)

    ''' Public methods for the GOAL '''
    def get_goal_loc(self): return self.g_loc;
    def post_goal(self, goal_loc):
        self._post(goal_loc, goalLayer)
        self.g_loc = goal_loc
    def put_goal(self, new_loc):
        self._del(self.g_loc, agentLayer)
        self.post_goal(new_loc)

    ''' Public methods for the IMMOBILE BLOCKS '''
    def isImBlocked(self, loc): 
        self.desparse()
        return self.grid[loc[XDIM], loc[YDIM], immobileLayer]
    def post_immobile_blocks(self, opts):
        self.desparse()
        if opts=='border':
            [self._post((x,y), immobileLayer) for x in range(self.gridsz[XDIM])\
                    for y in [0, self.gridsz[YDIM]-1] ]
            [self._post((x,y), immobileLayer) for x in [0, self.gridsz[XDIM]-1]\
                    for y in range(1, self.gridsz[YDIM]-1) ]
        elif type(opts)==list:
            for l in opts: self._post(l, immobileLayer);
        elif len(opts)==2:
            self._post(opts, immobileLayer);

    ''' Public methods for the MOBILE BLOCKS '''
    def isMoBlocked(self, loc): 
        self.desparse()
        return self.grid[loc[XDIM], loc[YDIM], mobileLayer]
    def post_mobile_blocks(self, locs_list):
        self.desparse()
        if type(locs_list)==list:
            for loc in locs_list: 
                self._post(loc, mobileLayer)
        elif len(locs_list)==2:
            self._post(locs_list, mobileLayer);
    def put_mobile_block(self, ploc, nloc):
        self._del(ploc, mobileLayer)
        self._post(nloc, mobileLayer)

    ''' Method: find out what kind of object is located at a queried location '''
    def getQueryLoc(self, loc, exception=False):
        self.desparse()
        if self.grid[loc[XDIM], loc[YDIM],immobileLayer]==1:return immobileLayer
        if self.grid[loc[XDIM], loc[YDIM], mobileLayer]==1:   return mobileLayer
        if self.grid[loc[XDIM], loc[YDIM], agentLayer]==1:    return agentLayer
        if self.grid[loc[XDIM], loc[YDIM], goalLayer]==1:     return goalLayer
        if not exception:
            print("FLAG 98")
        return -1 # <- flag for 'empty square'

    def getAllLocs(self, m_or_i):
        self.desparse()
        if m_or_i=='m':
          return [list(x) for x in np.argwhere(self.grid[:,:,mobileLayer]>0)]
        if m_or_i=='i':
          return [list(x) for x in np.argwhere(self.grid[:,:,immobileLayer]>0)]
            
    ''' Dump the entire grid representation, for debugging '''
    def dump_state(self):
        self.desparse()
        for i in range(NUM_LAYERS):
            print("Layer #"+str(i)+", the "+layer_names[i]+":")
            print(self.grid[:,:,i], '\n')

    ''' Return an identical but distinct version of this state object '''
    def copy(self):
        s_p = state(self.gridsz, self.name)
        s_p.grid = np.copy(self.grid)
        s_p.a_loc = self.a_loc
        s_p.g_loc = self.g_loc
        s_p.sparse = self.sparse
        return s_p

    ''' Use this function for testing equality between states. '''
    def equals(self, s_p):
        return s_p.sparse == self.sparse and self.gridsz==s_p.gridsz \
                and s_p.a_loc == self.a_loc and s_p.g_loc == self.g_loc \
                and np.array_equal(self.grid, s_p.grid) 

    ''' Use this function for testing equality between states. '''
    def equals2(self, s_p):
        return np.array_equal(self.grid, s_p.grid) 

    ''' Rotates a state by 90*rot degrees.  Only supported currently for 
        square grids.'''
    def rotate(self, rot): 
        raise Exception("Deprecated...?")
        aloc = self.a_loc;
        gloc = self.g_loc;
        if rot==0: 
            pass
        if rot==3:
            self.a_loc = (aloc[1], self.gridsz[0]-aloc[0]-1)
            self.g_loc = (gloc[1], self.gridsz[0]-gloc[0]-1)
        if rot==2:
            self.a_loc = (self.gridsz[0]-aloc[0]-1, self.gridsz[1]-aloc[1]-1)
            self.g_loc = (self.gridsz[0]-gloc[0]-1, self.gridsz[1]-gloc[1]-1)
        if rot==1:
            self.a_loc = (self.gridsz[1]-aloc[1]-1, aloc[0])
            self.g_loc = (self.gridsz[1]-gloc[1]-1, gloc[0])
        self.grid = np.rot90(self.grid, rot)

    ''' Get a boolean of whether or not I am a valid, consistent state. '''
    def checkValidity(self, long_version):
        self.desparse()
        if self.isImBlocked(self.a_loc)==1.0: return False
        if long_version: # <- check extra properties for new init. Bloated!!
            if self.isImBlocked(self.a_loc)==1.0: 
                print("FLAG 42")
                return False
            if self.isMoBlocked(self.a_loc)==1.0: 
                print("FLAG 25")
                return False
            for x in range(self.gridsz[XDIM]):
              for y in range(self.gridsz[YDIM]):
                if (self.grid[x, y, goalLayer] and \
                        self.grid[x, y, immobileLayer]): 
                    print("FLAG 95")
                    return False
                if (self.grid[x, y, goalLayer] and \
                        self.grid[x, y, mobileLayer]): 
                    print("FLAG 85")
                    return False
                if (self.grid[x, y, immobileLayer] and \
                        self.grid[x, y, mobileLayer]):
                    print("FLAG 50")
                    return False
                if ((x==0 or y==0 or x==self.gridsz[XDIM]-1 or y==self.gridsz[YDIM]-1)\
                        and self.grid[x, y, immobileLayer]==0.0): 
                    print("FLAG 05")
                    return False
        return True

    def dump_grid(self):
        if not self.sparse:
            for y in range(self.gridsz[1]):
                for i in range(NUM_LAYERS):
                    print(self.grid[:,y,i], '\t',)
                print('')
        else:
            print('{',)
            for i in range(NUM_LAYERS):
                if len(self.grid[i].data)==0: continue
                print(str(i)+':', self.grid[i].row, self.grid[i].col,)
                print(self.grid[i].data,',',)
            print('}')


class environment_handler3(object):
    '''
    class that handles objects [Rotational version]:
    manages states and actions on them and their logic
    '''
    def __init__(self, gridsize, action_mode, card_or_rot, \
            default_agent_dir=NORTH, default_world_dir=NORTH,\
            world_fill='roll', expt_name = None):
        self.gridsz = gridsize;
        assert(gridsize[0]==gridsize[1] or expt_name == 'gould-card-1')
        if 'egocentric'==action_mode and not world_fill in ['O','I','roll']:
            raise Exception("Please provide a valid fill for this environment"\
                            +"that facilitates map shifting.")
        self.allstates = []
        self.optDist = -1
        self.action_mode = action_mode
        self._AgentFwd = default_agent_dir
        self._WorldTop = default_world_dir
        self._WorldFill = world_fill # If the map shifts, place new: O or I

#    def Fwd(self): return self._AgentFwd + 0 % 4 # == self._AgentFwd
#    def Rgt(self): return self._AgentFwd + 1 % 4
#    def Bck(self): return self._AgentFwd + 2 % 4
#    def Lft(self): return self._AgentFwd + 3 % 4
    
    ''' Initialize a new state with post_state. '''
    def post_state(self, parameters, except_init=False, name=None):
        '''
        Convention: parameters should be a dict of:
            'agent_loc' = (x,y),  'goal_loc' = (x,y), 'immobiles_locs' in:
            {'borders' which fills only the borders, list of points}, 
            'mobiles_locs' = list of points.
        '''
        S = state(self.gridsz, name)
        S.post_agent(parameters['agent_loc'])
        S.post_goal(parameters['goal_loc'])
        S.post_immobile_blocks(parameters['immobiles_locs'])
        if 'mobiles_locs' in list(parameters.keys()):
            S.post_mobile_blocks(parameters['mobiles_locs'])
        # S.dump_state();
        if not except_init and not self.checkValidState(S, long_version=True):
            raise Exception("Invalid state attempted initialization: Flag 84")
        self.allstates.append(S)
        return S;
    
    ''' Helper: return a new location following an action '''
    def newLoc(self, loc, action):
        if (action in ['u','^',UDIR]): return (loc[XDIM],loc[YDIM]-1) # u and d are reversed
        if (action in ['d','v',DDIR]): return (loc[XDIM],loc[YDIM]+1) # because of file 
        if (action in ['r','>',RDIR]): return (loc[XDIM]+1,loc[YDIM])
        if (action in ['l','<',LDIR]): return (loc[XDIM]-1,loc[YDIM])
        if (action in rots): return (loc[XDIM],loc[YDIM])

    ''' Assertion: Verifies that a state is consistent '''
    def checkValidState(self, State, long_version=False):
        if State==None or not type(State)==state: return False
        return State.checkValidity(long_version)

    ''' Determines whether an action is valid given a state '''
    def checkIfValidAction(self, State, action) :
        if action==None:
            return True
        a_curloc = State.get_agent_loc();
        queryloc = self.newLoc(a_curloc, action)
        ''' First, verify that the action is on the map: '''
        if queryloc[XDIM]<0 or queryloc[XDIM]>=self.gridsz[XDIM] or \
                queryloc[YDIM]<0 or queryloc[YDIM]>=self.gridsz[YDIM]:
            return False
        ''' Second, check if the agent tries to push a block, that it is valid: '''
        if State.isMoBlocked(queryloc)==1.0:
            blockQueryloc = self.newLoc(queryloc, action)
            if State.isImBlocked(blockQueryloc)==1.0: return False
        ''' Third, check if the agent direction is not blocked by immobile:'''
        if State.isImBlocked(queryloc)==1.0: return False
        return True

    def rotate(self, state, rot): 
        state.rotate(rot); return state

    ''' Returns the State that should result from the actionID [see head of 
        file]. Based on this instance's specified action_mode, the action
        will do diffent things. 
        Specifically: return an 3-tuple of: 
            1. Move agent in which cardinal direction, regardless of world move 
            2. Move world in which cardinal direction, regardless of agent move
    '''

    def _get_action_from_actionID(self, ID):
        if self.action_mode == 'allocentric':
            return (cardinals[ID], NO_MOVE)
        if self.action_mode == 'egocentric':
            return (cardinals[ID], inv_cards[ID])

    ''' User-facing method: given 1/4 actions, return the updated state '''
    def performActionInMode(self, State, actionID, mode=None):
        try: action_mode = (self.action_mode if mode==None else mode)
        except:  raise Exception("No action mode specified.")
        action_tuple = self._get_action_from_actionID(actionID)
        return self._performGenericAction(State, action_tuple)

    ''' Given a tuple of (a_mv, w_mv, a_rot, w_rot), perform that update. '''
    def _performGenericAction(self, State, action_tuple):
        valid = self.checkValidState(State) and \
                self.checkIfValidAction(State, action_tuple[0])
        if not valid: return State # same state as before: current state.

        tmp_1 = self._shiftAgent(State.copy(), action_tuple[0])
        tmp_2 = self._shiftMap(tmp_1, action_tuple[1])
        return tmp_2


    ''' Given a world and direction, move only the Agent.'''
    def _shiftAgent(self, S, shift):
        al = agentLayer
        if self._WorldFill=='roll':
            if shift==UDIR: S.grid[:,:,al] = np.roll(S.grid[:,:,al], -1, axis=1)
            if shift==RDIR: S.grid[:,:,al] = np.roll(S.grid[:,:,al],  1, axis=0)
            if shift==DDIR: S.grid[:,:,al] = np.roll(S.grid[:,:,al],  1, axis=1)
            if shift==LDIR: S.grid[:,:,al] = np.roll(S.grid[:,:,al], -1, axis=0)
        elif self._WorldFill in ['O','I']:
            raise Exception("Not implemented; roll presumed 'better' for now.")
        return S

    ''' Given a world and direction, move the entire map, including Agent.'''
    def _shiftMap(self, S, shift):
        if self._WorldFill=='roll':
            if shift==UDIR: S.grid = np.roll(S.grid, -1, axis=1)
            if shift==RDIR: S.grid = np.roll(S.grid,  1, axis=0)
            if shift==DDIR: S.grid = np.roll(S.grid,  1, axis=1)
            if shift==LDIR: S.grid = np.roll(S.grid, -1, axis=0)
        elif self._WorldFill in ['O','I']:
            raise Exception("Not implemented; roll presumed 'better' for now.")
        return S

    def _rotMap(self, S, shift):
        pass

    ''' Returns the State that should result from the action.  if the action 
    was invalid, it returns the original state. newState: a bool flag for if 
    you want to overwrite the input State or return a new one, with the 
    original untouched.  '''
    def performAction(self, State, action, rot=0, newState=True):
        valid = self.checkValidState(State) and \
                self.checkIfValidAction(State, action)
        if not valid: return State # same state as before: current state.

        State_prime = State.copy() if newState else State
        if action=='no_move': # ie, only rotate
            return self.rotate(State_prime, rot)

        newAgentLoc = self.newLoc(State.get_agent_loc(), action)
        if State_prime.isMoBlocked(newAgentLoc)==1.0: 
            newBlockLoc = self.newLoc(newAgentLoc, action)
            State_prime.put_mobile_block(newAgentLoc, newBlockLoc)
        State_prime.put_agent(newAgentLoc)
        if rot>0 and not rot==ROT0:
            State_prime = self.rotate(State_prime, rot)
        return State_prime

    ''' utilities '''
    def isGoalReached(self, State):
        try:
            if State==None:
                return False;
        except: pass
        return State.sparse and State.grid[0]==State.grid[1] \
                or np.array_equal(State.grid[:,:,0], State.grid[:,:,1]) \
                or np.array_equal(State.get_agent_loc(), State.get_goal_loc())

    # TODO: update this V
    def getActionValidities(self, s): return np.array(\
        [ self.checkIfValidAction(s, a) for a in cardinals ], \
            dtype='float32')
    def getGridSize(self): return self.gridsz;

    '''  Debugging methods  '''
    def displayGameState(self, State=None, exception=True):
        ''' Print the state of the game in a visually appealing way. '''
        for y in range(self.gridsz[YDIM]):
            l = []
            for x in range(self.gridsz[XDIM]):
                flag = State.getQueryLoc((x,y), exception)
                if flag==agentLayer: l += '@'
                elif flag==goalLayer: l += 'X'
                elif flag==immobileLayer: l += '#'
                elif flag==mobileLayer: l += 'O'
                elif flag==-1: l += '-'
            print(' '.join(l))

    def displayTransition(self, S1, S2):
        for y in range(self.gridsz[YDIM]):
            l1 = []; l2=[]
            for x in range(self.gridsz[XDIM]):
                flag = S1.getQueryLoc((x,y), True)
                if flag==agentLayer: l1 += '@'
                if flag==goalLayer: l1 += 'X'
                if flag==immobileLayer: l1 += '#'
                if flag==mobileLayer: l1 += 'O'
                if flag==-1: l1 += '-'
                flag = S2.getQueryLoc((x,y), True)
                if flag==agentLayer: l2 += '@'
                if flag==goalLayer: l2 += 'X'
                if flag==immobileLayer: l2 += '#'
                if flag==mobileLayer: l2 += 'O'
                if flag==-1: l2 += '-'
            if not y==self.gridsz[YDIM]//2:
                print(' '.join(l1), '    ', ' '.join(l2))
            else:
                print(' '.join(l1), ' -> ', ' '.join(l2))
        print('')

    def printOneLine(self, State, mode='print', ret_Is=False):
        ''' Prints the state as succintly as possible '''
        s = ''; 
        s+= 'agent:'+str(State.get_agent_loc())+'\t'
        s+= ' goal:'+str(State.get_goal_loc())+'\t'
        Ms = State.getAllLocs('m')
        for m in Ms:
            s+= ' M:'+str(m)
        if len(Ms)==0:
            s+= ' [no M found]'
        s += '\t'
        if ret_Is:
          Is = State.getAllLocs('I')
          for i in Is:
            if (i[XDIM]>0 and i[XDIM]<State.gridsz[XDIM]-1 and \
                    i[YDIM]>0 and i[YDIM]<State.gridsz[YDIM]-1):
                s+= ' I:'+str(m)
          if len(Is)==0:
            s+= ' [no inner I found]'
        if mode=='print': print(s); 
        elif mode=='ret': return s;
        else: print("ERR tag 82")

    def getStateFromFile(self, filename, except_init='none'): 
        return self._read_state_file(filename, except_init)
    def _read_state_file(self, fn, except_init):
        '''
        _read_init_state_file: this function takes a file (sourced from main's 
        directory path) and processes it as an initialized state file.
        Format: first line is the X-size of the map, second line is the Y-size,
        and the following lines are the y'th grid squares, X-long, with key:
            A = agent    G = goal    I = immobile block    M = mobile block
        '''
        with open(fn, 'r') as f:
            self.gridsz = (int(f.readline()), int(f.readline()))
            #revgs = (int(f.readline()), int(f.readline()))
            #self.gridsz = (revgs[1], revgs[0])
            parameters = {}
            parameters['immobiles_locs']=[]
            parameters['mobiles_locs']=[]
            for y in range(self.gridsz[YDIM]):
                read_data = f.readline().strip()
                for x,c in enumerate(read_data):
                    if c=='I': parameters['immobiles_locs'].append((x,y))
                    if c=='M': parameters['mobiles_locs'].append((x,y))
                    if c=='A': parameters['agent_loc'] = (x,y)
                    if c=='G': parameters['goal_loc'] = (x,y)
        self.states = []
        if except_init=='except':
            return self.post_state(parameters, True, name=fn[:-4])
        else:
            return self.post_state(parameters, False, name=fn[:-4])

    ''' Functions for accessing the optimal minimum number of steps required 
    to achieve the goal.  Used for testing. '''
    def getOptimalNumSteps(self, s): return self.optDist
    def _dist(self,a,b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    def postOptimalNumSteps(self, s, manual_dist=-1):
        if manual_dist>0:
            self.optDist = manual_dist;
            return;
        if self._dist(s.get_goal_loc(), s.get_agent_loc()) == 1:
            self.optDist = 1;
            return;
        print("Dist calculator not implemented for this state.")
        return;

    def _l1_dist(self, a, b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    def noise_imm_blocks(self, s):
        for i in range(s.gridsz[0]):
            for j in range(s.gridsz[1]):
                query = s.getQueryLoc((i,j))
                if np.random.rand()<0.5:
                    if query in [ mobileLayer, immobileLayer]:
                        s.grid[i,j,query] = 1-s.grid[i,j,query] # flip it
        aloc = s.get_agent_loc()
        gloc = s.get_goal_loc()
        for loc in [ (aloc[0],aloc[1]+1), (aloc[0],aloc[1]-1), \
                (aloc[0]-1,aloc[1]), (aloc[0]+1,aloc[1])]:
            if loc[0]==gloc[0] and loc[1]==gloc[1]: continue
            s._post(loc, immobileLayer)
        
        return s






class state_generator(object):
    ''' State generator class. Difficulty classes:
        v1: Empty map, 1 step from goal.    Implemented? N
        v2: Empty map, N steps from goal.   Implemented? N
        v3: ...
    '''
    def __init__(self, gridsz=None): 
        # For now, FIX gridsz so that a network architecture need not change
        self.gridsz = gridsz;
        self.state_diffs = {}
        self.components = {}
        self.prefabs = [] # named list of prefabs

    ''' Component prefab reader. (use the flags at line tagged 90!) :
        Each component prefab is an 'uninitialized' component of a map,
        a particular arrangement of objects that are not spatially fixed but
        are reproducible for various purposes.  KEY:
        - XSZ and YSZ are the size of the rectangular hull of the component
        - ARR is the array of the component as an indiactor array: the 
            first two dims are XSZ and YSZ and the third is NUM_LAYERS=4.
        - Boolean symmetry indicators: 
            LR if the component is invariant over flipping over Y axis.
            UD if the component is invariant over flipping over X axis.
            90 if the component is invariant over 90 degree rotations.
            180 if the component is invariant over 180 degree rotations, ie
            has LR and UD symmetry.  ''' 
    def ingest_component_prefabs(self, pathname):
        component_files = [ f for f in os.listdir(pathname) \
                if os.path.isfile(os.path.join(pathname, f))]
        for c in component_files:
            slot = {}
            A = self._read_component(pathname, c)
            slot[ARR] = A
            slot[XSZ] = A.shape[0]
            slot[YSZ] = A.shape[1]
            R1 = np.rot90(A, 1, axes=(0,1))
            R2 = np.rot90(A, 2, axes=(0,1))
            name = os.path.splitext(c)[0]
            slot[FLIPUD] = np.array_equal(A, np.fliplr(A)) # * note:
            slot[FLIPLR] = np.array_equal(A, np.flipud(A)) # reversed!
            slot[ROT90]  = np.array_equal(A, R1) and R1.shape==A.shape
            slot[ROT180] = np.array_equal(A, R2) and R2.shape==A.shape
            alocs = np.argwhere(A[:,:,agentLayer])
            if alocs.shape[0]>=1:
                slot[ALOC] = alocs[0]
                if alocs.shape[0]>1: raise Exception("Invalid state, tag73")
            glocs = np.argwhere(A[:,:,goalLayer])
            if glocs.shape[0]>=1:
                slot[GLOC] = glocs[0]
                if glocs.shape[0]>1: raise Exception("Invalid state, tag72")
            self.components[name] = slot
            self.prefabs.append(name)

    def _read_component(self, pathname, component):
        with open(os.path.join(pathname,component), 'r') as c:
            c_sz = (int(c.readline()), int(c.readline()))
            prefab = np.zeros((c_sz[XDIM], c_sz[YDIM], NUM_LAYERS))
            for y in range(c_sz[YDIM]):
                read_data = c.readline()
                for x,obj in enumerate(read_data):
                    if obj in LAYER_MAP:
                        prefab[x,y,LAYER_MAP[obj]]=1.0
        return prefab

    def _readable_component(self, c, print_or_ret):
        cmpnt = self.components[c]
        s = c+': { sz('+str(cmpnt[XSZ])+','+str(cmpnt[YSZ])+'); ' 
        if np.any(cmpnt[ARR][:,:,agentLayer]): s+= 'A: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,agentLayer])).replace('\n','/')+'; '
        if np.any(cmpnt[ARR][:,:,goalLayer]): s+= 'G: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,goalLayer])).replace('\n','/')+'; '
        if np.any(cmpnt[ARR][:,:,immobileLayer]): s+= 'I: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,immobileLayer])).replace('\n','/')+'; '
        if np.any(cmpnt[ARR][:,:,mobileLayer]): s+= 'M: '+np.array_str(\
                np.argwhere(cmpnt[ARR][:,:,mobileLayer])).replace('\n','/')+'; '
        s+= 'Symms: '
        if cmpnt[FLIPLR]: s+= 'LR '
        if cmpnt[FLIPUD]: s+= 'UD '
        if cmpnt[ROT90]: s+= '90 '
        if cmpnt[ROT180]: s+= '180 '
        s+='}'
        if print_or_ret=='print': print(s)
        elif print_or_ret=='return': return s
            
    def generate_all_states_fixedCenter(self, version, env, oriented=False):
        if version=='v1': 
            return self._generate_v1('default_center', env, oriented)
    def generate_all_states_micro(self, version, env):
        if version=='v1': 
            return self._generate_micro('default_center', env)
    def generate_all_states_upto_2away(self, version, env):
        if version=='v2': 
            return self._generate_v2('default_center', env, 'leq')
    def generate_all_states_only_2away(self, version, env):
        if version=='v2': 
            return self._generate_v2('default_center', env, 'eq')

    def generate_all_states_floatCenter(self, version):pass
    def generate_N_states_fixedCenter(self, version, replacement=False):pass
    def generate_N_states_floatCenter(self, version, replacement=False):pass

    ''' -- verify where: assert that the requested [where] location fits in 
        the specified map size and in which the non-blocked-out space is of 
        dims (field_x, field_y). '''
    def _verifyWhere(self, where, field_sz):
        field_x, field_y = field_sz
        if not (len(where)==2 and where[XDIM]>=1 and where[YDIM]>=1 and \
                where[XDIM]+field_x < self.gridsz[XDIM] and \
                where[YDIM]+field_y < self.gridsz[YDIM] ):
            print("Invalid center for v1 init:", where)
            return None
        return where
    
    def _adjust_for_orientation(self, whichCmp, cmpOrien, field_sz, cmpLoc):
        ''' Rearrange the component's matrix based on requested orientation.'''
        cmpnt = self.components[whichCmp]
        c_arr = cmpnt[ARR]
        if cmpOrien[0]: c_arr = np.fliplr(c_arr)
        if cmpOrien[1]: c_arr = np.flipud(c_arr)
        c_arr = np.rot90(c_arr, cmpOrien[2], axes=(0,1))
        arr = np.zeros((field_sz[XDIM], field_sz[YDIM], NUM_LAYERS))
        if cmpOrien[2]%2==0:
            arr[ cmpLoc[XDIM] : cmpLoc[XDIM] + cmpnt[XSZ], \
                cmpLoc[YDIM] : cmpLoc[YDIM] + cmpnt[YSZ], :] = c_arr
        else:
            arr[ cmpLoc[XDIM] : cmpLoc[XDIM] + cmpnt[YSZ], \
                cmpLoc[YDIM] : cmpLoc[YDIM] + cmpnt[XSZ], :] = c_arr
        return arr

    def _initialize_component(self, template, rootloc, field_sz, \
                  whichCmp, cmpLoc, cmpOrien=(0,0,0), mode='arr'):
        ''' Enrichens the template via masking. No verifications! 
            cmpLoc: *relative* to the root location.
            cmpOrien: orientation of (flipUD?, flipLR?, 90*x rotation?)'''

        # Create new parameters dict that interfaces the state system
        if template==None and mode=='param':
            parameters = {}
            parameters['mobiles_locs'] = []
            parameters['immobiles_locs'] = []
            self._fill_Immobiles_dict(parameters, field_sz, rootloc)

        arr = self._adjust_for_orientation(whichCmp, cmpOrien, field_sz, cmpLoc)
        

        if mode=='param':
            # Set the (single) agent parameter
            alocs = np.argwhere(arr[:,:,agentLayer])
            if alocs.shape[0]==1: aloc = alocs[0]
            aloc[XDIM] += rootloc[XDIM]# + cmpLoc[XDIM] <- mystery fix...
            aloc[YDIM] += rootloc[YDIM]# + cmpLoc[YDIM]
            parameters['agent_loc'] = aloc

            # Set the (single) goal parameter
            glocs = np.argwhere(arr[:,:,goalLayer])
            if glocs.shape[0]==1: gloc = glocs[0]
            gloc[XDIM] += rootloc[XDIM]# + cmpLoc[XDIM]
            gloc[YDIM] += rootloc[YDIM]# + cmpLoc[YDIM]
            parameters['goal_loc'] = gloc

            # Set the immobile block parameters
            if not 'immobiles_locs' in parameters:
                parameters['immobiles_locs']=[]
            ilocs = np.argwhere(arr[:,:,immobileLayer])
            for iloc in ilocs:
                iloc[XDIM] += rootloc[XDIM] + cmpLoc[XDIM]
                iloc[YDIM] += rootloc[YDIM] + cmpLoc[YDIM]
                parameters['immobiles_locs'].append(iloc)

            # Set the mobile block parameters
            if not 'mobiles_locs' in parameters:
                parameters['mobiles_locs']=[]
            mlocs = np.argwhere(arr[:,:,mobileLayer])
            for mloc in mlocs:
                mloc[XDIM] += rootloc[XDIM] + cmpLoc[XDIM]
                mloc[YDIM] += rootloc[YDIM] + cmpLoc[YDIM]
                parameters['mobiles_loc'].append(mloc)

            return parameters

        elif mode=='arr':
            template[rootloc[XDIM] : rootloc[XDIM] + field_sz[XDIM], \
                     rootloc[YDIM] : rootloc[YDIM] + field_sz[YDIM], :] = arr
            return template

    '''
        Convention: parameters should be a dict of:
            'agent_loc' = (x,y),  'goal_loc' = (x,y), 'immobiles_locs' in:
            {'borders' which fills only the borders, list of points}, 
            'mobiles_locs' = list of points.
    '''
    def _fill_Immobiles_dict(self, parameters, field_shape, centerloc):
        ''' _fill_immobiles but with parameter dict retval '''
        parameters['immobiles_locs']=[]
        for a in range(self.gridsz[XDIM]):
            for b in range(self.gridsz[YDIM]):
                if a==0 or b==0 or a==self.gridsz[XDIM] or b==self.gridsz[YDIM] \
                    or a>=field_shape[XDIM]+centerloc[XDIM] \
                    or b>=field_shape[YDIM]+centerloc[YDIM]:
                        parameters['immobiles_locs'].append((a,b))
        return parameters

    def _generate_micro(self, rootloc, env):
        ''' V1 EXTRA easy. These states are 3x3 and place the agent directly 
        next to the goal AND block all other directions for agent. '''
        field_shape = (3,3) # for all V1 states.
        if rootloc=='default_center': rootloc=(1,1)
        if not self._verifyWhere(rootloc, field_shape): sys.exit()
        sp_u = self._initialize_component(None, rootloc, field_shape,\
                'nextto_force', (0,0), (0,0,0), 'param')
        sp_r = self._initialize_component(None, rootloc, field_shape,\
                'nextto_force', (0,0), (0,0,1), 'param')
        sp_d = self._initialize_component(None, rootloc, field_shape,\
                'nextto_force', (0,0), (0,0,2), 'param')
        sp_l = self._initialize_component(None, rootloc, field_shape,\
                'nextto_force', (0,0), (0,0,3), 'param')
        states = [env.post_state(sp) for sp in [sp_u, sp_r, sp_d, sp_l]]
        return states


    def _generate_v1(self, rootloc, env, oriented):
        ''' V1, version one easiest state. These states are 3x3 and place the 
        agent directly next to the goal. '''
        field_shape = (3,3) # for all V1 states.
        if rootloc=='default_center': rootloc=(1,1)
        if not self._verifyWhere(rootloc, field_shape): sys.exit()
        states = []
        for x in range(field_shape[XDIM]-1):
            for y in range(field_shape[YDIM]):
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,0,0), 'param')
                states.append(env.post_state(sp))
                if oriented: continue
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,1,0), 'param')
                states.append(env.post_state(sp))
        for y in range(field_shape[YDIM]-1):
            for x in range(field_shape[XDIM]):
                if oriented: continue
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,0,1), 'param')
                states.append(env.post_state(sp))
                sp = self._initialize_component(None, rootloc, field_shape,\
                        'nextto', (x,y), (0,1,1), 'param')
                states.append(env.post_state(sp))
        return states
    # exclusion: 1-away and 2-away (leq), or just 2-away?
    def _generate_v2(self, rootloc, env, exclusion='leq', Dir=None):
        if Dir==None:
            Dir = './data_files/states/'
        if not rootloc=='default_center':
            raise Exception("rootloc not yet implemented : "+str(rootloc))
        if not env.gridsz==(7,7):
            raise Exception("env gridsz not yet implemented : "+str(env.gridsz))
        tag_keep = '7x7-2away-A-'
        if exclusion=='eq': tag_remove = '7x7-2away-A-_'
        elif exclusion=='leq': tag_remove = 'sentinel dont remove me'
        files = [fn for fn in os.listdir(Dir) if tag_keep in fn \
                and not tag_remove in fn]
        files.sort()
#        for fn in files:
#            print(fn[12:-4],)
#        print('')
        return [env.getStateFromFile(os.path.join(Dir,fn), 'except') \
                for fn in files]





def test_script1():
    # this script tests the ability to generate all game states for vers1.
    #foo = state_generator((10,10))
    #env = environment_handler((10,10))

    foo = state_generator((5,5))
    env = environment_handler3((5,5))
    foo.ingest_component_prefabs("./data_files/components/")
    X = foo.generate_all_states_fixedCenter('v1', env)

    for i,s in enumerate(X):
        print(i,':')
        env.displayGameState(s); print('')
        env.postOptimalNumSteps(s)
    print("Min number of steps to solve above game: ", env.getOptimalNumSteps(random.choice(X)))
    print("Number of states generated:", len(X))
    print("Above are all the possible valid game states that have a 3x3",)
    print(" grid in a fixed location in which the agent is directly next",)
    print(" to the goal (in any direction); that is, the first possible task.")
    print('--------------------------------------------------------')

def test_script2():
    # this script tests the ability to generate all game states for vers1.
    foo = state_generator((10,10))
    env = environment_handler((10,10))
    foo.ingest_component_prefabs("./data_files/components/")
    X = foo.generate_all_states_fixedCenter('v1', env, oriented=True)

    for i,s in enumerate(X):
        env.displayGameState(s);
        env.postOptimalNumSteps(s)
    print("Min number of steps to solve above game: ", env.getOptimalNumSteps(random.choice(X)))
    print("Number of states generated:", len(X))
    print("Above are all the possible valid game states that have a 3x3",)
    print(" grid in a fixed location in which the agent is directly next",)
    print(" to the goal (in any direction); that is, the first possible task.")
    print('--------------------------------------------------------')


def test_script3():
    # This example script demonstrates the ability for a 
    print("The following is a test example.  For reference, @ is the agent, X is"+\
      " the goal, O is a movable block, and # is an immovable block.")

    for fn in ["./data_files/states/3x3-diag+M.txt"]:#, "./data_files/states/3x4-diag+M.txt"]:
      for act_seq in [ [('d',2), ('l',1)], [('d',1), ('r',2), ('r',3), ('l',0)]]:
        env = environment_handler3()
        s0 = env.getStateFromFile(fn)
        print("\nvalid initial state:", not s0==None);
        env.displayGameState(s0); 
        s0.dump_grid()
        #for action in [ ('d',0), ('d',0), ('r',0), ('u',0) ]:
        for action in act_seq:
            a, r = action
            s1 = env.performAction(s0, a,r);  print("\naction:",a,"action success:", \
                env.checkIfValidAction(s0, a));  
            print("Goal reached?: ", env.isGoalReached(s1))
            env.displayGameState(s1);
            s1.dump_grid()
            s0=s1
            

#        s1 = env.performAction(s0, 'l',3);  print "\naction: d, action success:", \
#            env.checkIfValidAction(s0, 'd');  env.displayGameState(s1);
#        s2 = env.performAction(s1, 'r',0);  print "\naction: r, action success:", \
#            env.checkIfValidAction(s1, 'r');  env.displayGameState(s2);
#        s3 = env.performAction(s2, 'u',0);  print "\naction: u, action success:", \
#            env.checkIfValidAction(s2, 'u');  env.displayGameState(s3);
        print('--------------------------------------------------------')
    print("Above is a test script that demonstrates that the state-environment-actor ")
    print("situation is coherent and functional.")


def test_script4():
    for mode in ['egocentric', 'allocentric']:
        print('--------------------------------------------------------')
        env = environment_handler3((5,5), mode, world_fill='roll')
        s0 = env.getStateFromFile('./data_files/states/3x3-basic-GU.txt', 'except')
        env.displayGameState(s0, exception=True)
        for mv in [RDIR, RDIR, RDIR, UDIR, RDIR, DDIR]:
            s1 = env.performActionInMode(s0, mv)
            env.displayGameState(s1, exception=True)
            s0=s1

def test_script5():
    # this script tests new 2-away maps.
    Dir = './data_files/states/'
    for mode in ['egocentric', 'allocentric']:
        env = environment_handler3((7,7), mode, world_fill='roll')
        states = [env.getStateFromFile(os.path.join(Dir,fn), 'except') \
                for fn in os.listdir(Dir) if '7x7-2away-A' in fn]
        for s in states:
            env.displayGameState(s, exception=True)
            print('')



if __name__=='__main__':
    #test_script1()
    #test_script2()
    #test_script3()
    #test_script4()
    test_script5()
    print("DONE")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# [formerly independent expenv.py]


# The point of this script is to help me develop on 
# issues that I still need to identify since it's been
# almost two months since I felt the code.
#
# Task: Run a simple curriculum learning experiment on 
# the flavor-place location task by Tse (2007): that is,
# with four starting locations, one of six signals, and
# six corresponding possible targets. Mimic 'digging'
# by placing a block over each starting location.

# General utilities:
def map_nparr_to_tup(Iterable):
    return tuple([value.tolist()[0] for value in Iterable])

#def addvec(Iterable, m, optn=None):
#    try: 
#        return tuple([i+m for i,m in zip(Iterable,m)])
#    except:
#        return tuple([i+m for i in Iterable])
def addvec(Iterable, m, optn=None):
    try: 
        m[0]
    except:
        if m>80: 
            m = DVECS[m]
        else:
            m = DVECS[INDICES_TO_CARDINAL_ACTIONS[m]]
    return tuple([i+m for i,m in zip(Iterable,m)])

def multvec(Iterable, m, optn=None):
    if optn=='//':  return tuple([i//m for i in Iterable])
    if optn=='/':   return tuple([i/m for i in Iterable])
    if optn==int:   return tuple([int(i*m) for i in Iterable])
    return tuple([i*m for i in Iterable])

def at(mat, pos, lyr): return mat[pos[X], pos[Y], lyr]
def empty(mat, pos): return np.any(mat[pos[X], pos[Y], :])
def what(mat, pos): return np.array([at(mat, pos, lyr) for lyr in OLAYERS])
def put(mat, pos, lyr, v): mat[pos[X], pos[Y], lyr] = v
def put_all(mat, pos_list, lyr, v):
    for p in pos_list: put(mat, p, lyr, v)

#------#------#------#------#------#------#------#------#------#------#------#--
#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*
#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*
#
#   Experiment API class:
#
#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*$#*
#------#------#------#------#------#------#------#------#------#------#------#--


# Experiment class:
class ExpAPI(environment_handler3):
    def __init__(self, experiment_name, centr, card_or_rot='card', debug=False):
        ''' Initializer for Experiment class. Please provide:
         - experiment_name keyword, which indirectly signals the starting 
            states. Currently takes 'tse2007'.
         - centr, the keyword for the reference frame. Currently takes 
            'allocentric' or 'egocentric' but in future will facilitate 
            rotational heading frames.
         - optional debug-mode boolean flag
        ''' 
        environment_handler3.__init__(self, gridsize = \
                { 'tse2007': (11,11), 'r-u': (7,7), 'ru': (7,7), \
                'r-u-ru': (7,7), 'gould-card-1':(5,3) }[experiment_name], \
                action_mode = centr, card_or_rot=card_or_rot, expt_name = \
                experiment_name)
        self.centr = centr
        self.card_or_rot = card_or_rot
        self.state_gen = state_generator(self.gridsz)
        self.start_states = []
        if not experiment_name=='gould-card-1':
            self._set_starting_states({\
                    'tse2007':TEMPLATE_TSE, \
                    'r-u-ru': TEMPLATE_R_U_RU, \
                    'r-u': TEMPLATE_R_U, \
                    }[experiment_name], debug)
        else:pass
        self.experiment_name = experiment_name

    def _find_all(self, a_str, char):
        # [internal] scan from a template string (eg, TEMPLATE_TSE)
        s = a_str.replace(' ','')
        startX, startY = 0,0
        for c in s:
            if c==char: 
                yield((startX, startY))
            elif c=='r': 
                startY += 1
                startX = 0
            if c in 'a!xm.': 
                startX += 1

    # Set this experiment's possible starting states using complete template str
    def _set_starting_states(self, state_template, debug=False):
        oind = state_template.index('o')
        if state_template.index('e') > oind: raise Exception()
        num_start_locs = state_template.count('a')
        num_goal_locs = state_template.count('!')
        if not state_template.find('*') > oind: raise Exception()

        start_locs = list(self._find_all(state_template, 'a'))
        goal_locs = list(self._find_all(state_template, '!'));
        block_locs = list(self._find_all(state_template, 'x'));
        if 'D' in state_template:
            mobile_locs = list(self._find_all(state_template, '!'));
            self.valid_states = np.array( [AL, GL, AL|GL, IL, ML, ML|GL] ).T
        else:
            try:
                mobile_locs = list(self._find_all(state_template, 'm'));
            except:
                mobile_locs = []
                self.valid_states = np.array( [AL, GL, AL|GL, IL, ML] ).T
#        self.valid_states = np.append(self.valid_states, np.expand_dims(\
        #                np.array([0,0,0,0], dtype=bool)), axis=0)

        rx = [0,1,self.gridsz[X]-2, self.gridsz[X]-1]
        ry = [0,1,self.gridsz[Y]-2, self.gridsz[Y]-1]

        ''' flavor == goal here. '''
        for start_box in start_locs:
            for flav_id, flavor_loc in enumerate(goal_locs):
                st = np.zeros( (self.gridsz[X], self.gridsz[Y], NUM_LAYERS))
                put(st, start_box, agentLayer, True)
                put(st, flavor_loc, goalLayer, True)
                put_all(st, mobile_locs, mobileLayer, True)
                put_all(st, block_locs,  immobileLayer, True)

                self.start_states.append( { 'flavor signal': flav_id, \
                        'state': st, '_whichgoal':flav_id, \
                        '_startpos':start_box, 'goal loc':flavor_loc })
                #        rnd_state = self.start_states[np.random.choice(range(24))]


        self.curr_sorted_states = self.start_states.copy()
        def dist(state_):
            x,y = state_['goal loc'], state_['_startpos']
            return abs(x[0]-y[0])+abs(x[1]-y[1])
        self.curr_sorted_states.sort(key=dist)


        rnd_state = np.random.choice(self.start_states)
        if debug: 
            print('flag 93747')
            print_state(rnd_state, 'condensed')

    def _view_state_copy(self, st):
        sret = {}
        for key in ('_startpos','flavor signal','_whichgoal'):
            sret[key] = st[key]
        sret['state'] = np.copy(st['state'])
        return sret

    def get_random_starting_state(self): 
        ''' Public method: get a random state struct with fields: 'state', 
        '_startpos', 'flavor signal', '_whichgoal', 'goal loc'. The three 
        later fields are helper attributes for, say, curricula or presentation.   
        '''
        #st = self.start_states[np.random.choice(range(24))]
        return self._view_state_copy(np.random.choice(self.start_states))


    def get_weighted_starting_state(self, envir, pct):
        pct=float(pct)
#        print(envir,pct, 0.5*pct,1-pct)
        if envir=='r-u': raise Exception()
        assert (envir=='r-u-ru')
        ps = [0.5*pct, 0.5*pct, 1-pct]
        return self._view_state_copy(np.random.choice(\
                self.curr_sorted_states, p=ps))

#        for s in self.start_states:
#            print([s[x] for x in ['goal loc','_startpas'] ])
#        for s in self.curr_sorted_states:
#            print([s[x] for x in ['goal loc','_startpos'] ])


    def get_starting_state(self, curriculum_name, epoch, envir=None): 
        # interface wrapper method for submethods
        curr = curriculum_name
        cspl = curriculum_name.split(':')
        if curr==None:   return self.get_random_starting_state()['state']
        elif len(curr)>4 and curr[:4]=='FLAT' and len(cspl)==2:
            return self.get_weighted_starting_state(envir, float(cspl[1]))['state']
        elif len(curr)>6 and curr[:6]=='STEP-1' and len(cspl)==4:     
            if epoch >= int(cspl[3]):
                return self.get_weighted_starting_state(envir, cspl[2])['state']
            else:
                return self.get_weighted_starting_state(envir, cspl[1])['state']
        elif len(curr)>8 and curr[:8]=='LINEAR-1' and len(cspl)==4:
            param = min(1.0, max(0.0, epoch/float(cspl[3])))
            pct = param*float(cspl[2])+(1-param)*float(cspl[1])
            return self.get_weighted_starting_state(envir, pct)['state']
        else:
            raise exception(curr, cspl, epoch, envir) 

        return curriculum_name, 'error expenv line ~200'

        # Very hacky:
        assert(self.experiment_name=='r-u-ru')
        l1 = len(TEMPLATE_R_U)
        l2 = len(TEMPLATE_RU)
        if curriculum_name=='STEP':
            ps = [0.5, 0.5, 0] if False else False
        return self._view_state_copy(np.random.choice(self.start_states), p=ps)['state']

    def get_all_starting_states(self):
        ''' Public method: get a random state struct with fields: 'state', 
        '_startpos', 'flavor signal', '_whichgoal', 'goal loc'. The three 
        later fields are helper attributes for, say, curricula or presentation.   
        '''
        return [self._view_state_copy(st) for st in self.start_states]

    def get_agent_loc(self,state):    
        '''Public method: query the location of the agent. (<0,0> is NW corner.)'''
        return self._get_loc(state,targ='agent')

    def get_goal_loc(self,s):     return self._get_loc(s,targ='goal')
#    def get_allo_loc(self,s):     return self._get_loc(s,targ='map') # center

    def _get_loc(self, state_matrix, targ):
        if targ=='agent': 
            return map_nparr_to_tup(np.where(state_matrix[:,:,agentLayer]==1))
        if targ=='goal': 
            return map_nparr_to_tup(np.where(state_matrix[:,:,goalLayer]==1))
        if targ=='map':
            return multvec(self.gridsz, 2, '//') # center

    def _out_of_bounds(self, pos):
        return (pos[X] < 0 or pos[X] >= self.gridsz[X] or \
                pos[Y] < 0 or pos[Y] >= self.gridsz[Y])

    def _is_valid_move(self, st, move): 
        aloc = self.get_agent_loc(st)
        newaloc = addvec(aloc, move)
        if self._out_of_bounds(newaloc): return False
        if at(st, newaloc, immobileLayer): return False
        if at(st, newaloc, mobileLayer):
            st2 = np.copy(st)
            put(st2, newaloc, agentLayer, True)
            put(st2, aloc, agentLayer, False)
            return self._is_valid_move(st2, move)
        return True

    def _move_ent_from_to(self, mat, loc, nextloc, lyr):
        m2 = np.copy(mat)
        if not at(m2,loc,lyr): raise Exception()
        #print ("Adjusting",lyr,loc,nextloc)
        put(m2,loc,lyr, False)
        put(m2,nextloc,lyr, True)
        return m2

    def _adjust_blocks(self, mat, loc, dir_vec, debug=True):
        nloc = addvec(loc, dir_vec)
        if self._out_of_bounds(nloc): return mat, False
        arr = [what(mat, loc), what(mat, nloc)]
        ploc=nloc
        while True:
            nloc = addvec(ploc, dir_vec)
            #print('>>',dir_vec)
            if self._out_of_bounds(nloc): return mat, False
            if not arr[-1][mobileLayer]: return mat, not arr[-1][immobileLayer]
            nmat = self._move_ent_from_to(mat, ploc, nloc, mobileLayer)
            if len(arr)>2: put(nmat, ploc, mobileLayer, True)
            arr.append(what(mat, nloc))
            ploc=nloc
            mat=nmat
        raise Exception()

    def _move_agent(self, state_mat, dir_vec, ret_valid_move):
        aloc = self.get_agent_loc(state_mat)
        newL = addvec(aloc, dir_vec)
        state_mat2, success = self._adjust_blocks(state_mat, aloc, dir_vec)
        state_mat2 = self._move_ent_from_to(state_mat2, aloc, newL, agentLayer)
        if self.centr == 'egocentric':
            shft, axis = { 0:(1,1), 1:(-1,1), 2:(-1,0), 3:(1,0) }[dir_vec]
            state_mat2=np.roll(state_mat2, shift=shft, axis=axis)
        elif not self.centr == 'allocentric': raise Exception(self.centr)

        isValid = self._is_valid_move(state_mat, dir_vec)
        if isValid: 
            if ret_valid_move==True: return state_mat2, isValid
            else: return state_mat2
        if ret_valid_move==True: return state_mat, isValid
        return state_mat

    def _rot_agent(self, state_mat, nrots, ret_valid_move):
        aloc = self.get_agent_loc(state_mat)
        assert(self.experiment_name == 'r-u-ru')
        assert(nrots in [1,2,3])
        state_mat = np.rot90(state_mat, k=nrots, axes=(0,1))
#        state_mat2=np.roll(state_mat2, shift=shft, axis=axis)
        if self.centr == 'egocentric':
            centr_pos = (3,3)
            dx, dy = aloc[0]-centr_pos[0], aloc[1]-centr_pos[1]
            state_mat = np.roll(state_mat, shift=dx, axis=0)
            state_mat = np.roll(state_mat, shift=dy, axis=1)
        elif not self.centr == 'allocentric': raise Exception(self.centr)
        return state_mat, True


    def new_statem(self, orig_state, action, valid_move_too=False):
        '''Public Method: error-proofed public method for making (S') from (S,A)
        NOT currently errorproofed against egocentrism!'''
        if action>=100:
            return self._move_agent(orig_state, DVECS[action], valid_move_too)
        else:
            if self.card_or_rot=='card':
                return self._move_agent(orig_state, action, valid_move_too)
            elif self.card_or_rot=='rot':
                if action==0:
                    return self._move_agent(orig_state, 0, valid_move_too)
                else:
                    return self._rot_agent(orig_state, action, valid_move_too)



def _____dont_do_this__stub():
    for centr in ['egocentric', 'allocentric']:
        ExpAPI('tse2007', centr)._set_starting_states(TEMPLATE_TSE)



# fun little test script:
def deprecated1():
  if __name__=='__main__':
    ex = ExpAPI('tse2007', 'egocentric')
    cur_state = ex.get_random_starting_state()['state']
    while False:#True:
        print('current state:') 
        print('flag 36351')
        print_state(cur_state, 'condensed')
        print('current location:', ex.get_agent_loc(cur_state))
        inp = input(' interface input >> ')
        if not len(inp)==1: break
        try:
            inp_to_mov = {\
                    'N': MOVE_NORTH,
                    'S': MOVE_SOUTH,
                    'E': MOVE_EAST,
                    'W': MOVE_WEST, }[inp.upper()]
        except: 
            break
        next_state = ex.new_statem(cur_state, inp_to_mov)

        cur_state = next_state



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# [formerly PathfinderEnv.py]
'''
This module simply wraps the environment above for use in newer modules, 
especially openai's gym, which however we are not using anymore.
'''


def print_state(start_state, mode='condensed', print_or_ret='print'):
    S = ''
    if type(start_state)==np.ndarray:
        st = start_state
    else:
        st = start_state['state']
        S += str(mode+':')
    if mode=='matrices':
        for i in range(st.shape[-1]):
            S += str(st[:,:,i])
    if mode=='condensed':
        for y in range(st.shape[Y]):
            for x in range(st.shape[X]):
                if st[x,y,goalLayer] and st[x,y,agentLayer]: S += str('!')
                elif st[x,y,agentLayer]: S += str('I')
                elif st[x,y,goalLayer] and st[x,y,mobileLayer]: S += str('@')
                elif st[x,y,goalLayer]: S += str('*')
                elif st[x,y,immobileLayer]: S += str('-')
                elif st[x,y,mobileLayer]: S += str('o')
                elif 0==np.sum(st[x,y,:]): S += str(' ')
                else: 
                    S += str('#')
                    print(S)
#                raise Exception("Error", st[x,y,:],S)
            S += str('\n')
    if not type(start_state)==np.ndarray:
        S += str("Flavor signal/goal id: ", start_state['flavor signal'])

    if print_or_ret=='print': print(S)
    else: return S

def pr_st(st):
    return print_state(st, 'condensed', '')



#class PathEnv(gym.Env):
class PathEnv(object):
    ''' class PathEnv: an openai gym-compliant wrapper for a pathfinder 
            experimental environment. Please provide a well-instantiated 
            ExpAPI (a class that facilitates easy interaction with the 
            core environment objects). Feed this ExpAPI all the experimental
            parameters; this class is strictly just a wrapper.

        Methods defined here: nothing of note. 
    '''
    def __init__(self, exp_env, envir):
        sz = exp_env.getGridSize()
        self.action_space = Discrete(4)
        self.observation_space = spaces.Box(0,1, (sz[0], sz[1], NUM_LAYERS))
        self.envir=envir
                                       
        # ^^ Because gym Tuples don't come with one....!?

        self.current_state = exp_env.get_random_starting_state()['state']
        self.previous_state = self.current_state
        self.exp_env = exp_env        
        self.metadata = {'render.modes':['human','ansi','PRINT','NOPRINT']}
        self.reward_range = (0,1)
        self.flag=True

        if Config.GAME_NAME=='r-u-ru':
            if Config.CURRICULUM_NAME=='FLAT':
                pass#self.level_1_task = 'r-u-ru'
            elif Config.CURRICULUM_NAME in ['LIN','STEP']:
#                self.level_2_task = 'ru'
                self.levels = ['r-u', 'ru']
        print("a pathfinder environment wrapper has been imported")

    def reset(self, epoch=-1, t=-1, curr=None, test_train='train'): 
        if test_train=='test':
            return self._reset(test_train='test')
        if t<0 and epoch>=0: t=epoch
        if not curr==None:
            curriculum=curr
        elif self.flag and not self.exp_env.experiment_name=='r-u-ru':
            print("Flat curriculum used."); 
            self.flag=False;
            curriculum=None
        else:
            curriculum=Config.CURRICULUM_NAME 
        return self._reset(curriculum, t, test_train='train') # ugh might be buggy watch out





    def _reset(self, curriculum=None, t=-1, test_train='train'): 
        ''' Todo: augment so that it can take epoch args (which it 
            passes to the wrapped env for sampling. '''
#        if test_train=='train':
#            self.previous_state = self.current_state = \
#                        self.exp_env.get_starting_state(curriculum, t, envir=self.envir)
#        elif test_train=='test':
        if test_train=='test':
            self.previous_state = self.current_state = \
                    self.exp_env.get_random_starting_state()['state']
        elif test_train=='train':
            self.previous_state = self.current_state = \
                    self.exp_env.get_starting_state(curriculum, t, envir=self.envir)
        else: raise Exception(test_train)
                    
        return self.current_state
#        if curriculum==None:
#            self.current_state = self.exp_env.get_random_starting_state()['state']
#        else:
#            self.current_state = self.exp_env.get_starting_state_by_epoch()['state']
#        return self.current_state 
    
    def _render(self, mode=None, close=None): 
        #raise Exception("Debug. 442028 Mode: ", mode) This was inconclusive aaa deleted print several lines down.
        p= print_state(self.current_state, \
                           'condensed', 'string_ret')
        if mode in ['human','PRINT']: 
            print(p)
            return p
        elif mode=='NOPRINT': 
            return p
        else: raise Exception(mode, 'render mode not defined')

    def sample(self): 
        return random.choice(list(DVECS.keys()))

    def _step(self, actn):
        new_st, succ = self.exp_env.new_statem(\
            self.current_state, actn, valid_move_too=True)
        goalReached = self.exp_env.get_agent_loc(new_st) == \
                      self.exp_env.get_goal_loc(new_st)
        self.previous_state = self.current_state
        self.current_state = new_st
        return new_st, int(goalReached), (not succ) or goalReached, {}
    
    def get_num_actions(self): return len(DVECS) # untested


class PathEnvAuto(PathEnv):
    # Running this simply initializes the pathfinder environment according
    # to the config parameters
    def __init__(self):
        #super(PathEnv.__init__(self, exp_env(Config.GAME_NAME, Config.CENTRISM)))
        try:
            PathEnv.__init__(self, ExpAPI(Config.GAME_NAME, Config.CENTRISM))
        except:
            raise Exception()
                       
                                          

                       
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# [formerly custom_models.py]
'''
This module describes some network topologies. They are connected to
especially openai's gym, which however we are not using anymore.
'''



## this isn't working...   kept for reference
#def _orig_pf(inpt, num_actions, scope, seed=False, reuse=False):
#    with tf.variable_scope(scope, reuse=reuse):
#            if seed: 
#                scope.reuse_variables()
#                tf.reset_default_graph()
#                tf.set_random_seed(seed)
#                print( tf.get_seed)
#            out = inpt
#            out = layers.fully_connected(out, num_outputs=64)
#            out = tf.nn.relu(out)
#            out = tf.nn.dropout(out, 0.5)
#            out = layers.fully_connected(out, num_outputs=num_actions)
#            out = tf.nn.tanh(out)
#            return out
#
##def original_pathfinder_model(seed=None):
##    """This model is a recreation, as close as possible, to the original 
##    networks used for the results in Pathfinding Navigation.
##
##    Parameters
##    ----------
##    None. 
##
##    Returns
##    -------
##    q_func: function
##        q_function for DQN algorithm.
##    """
##    return lambda *args, **kwargs: _mlp(hiddens=[64], seed=seed, *args, **kwargs)
#
    

def original_pathfinder_model(config):
    """This model is a recreation, as close as possible, to the original 
    networks used for the results in Pathfinding Navigation.

    Parameters
    ----------
    None. 

    Returns
    -------
    function which takes as arguments (input_var, config, **kwargs)
    """
#    def _mlp_custom(hiddens, inpt, num_actions, scope, seed_int, \
#                    reuse=tf.AUTO_REUSE, layer_norm=False,config=None):
#        with tf.variable_scope(scope, reuse=reuse):
#            with tf.device(config._DEVICE): # Require a config!
#                tf.set_random_seed(seed_int)
#                out = tf.contrib.layers.flatten(inpt)
#                for hidden in hiddens:
#                    out = layers.fully_connected(out, num_outputs=hidden)
#                    out = tf.nn.dropout(out, 0.5)
#                    if layer_norm:
#                        out = layers.layer_norm(out, center=True, scale=True)
#                q_out = layers.fully_connected(out, num_outputs=num_actions,\
#                            activation_fn=tf.nn.tanh)
#                return q_out
#
    def _mlp_custom(inpt, c):
        if c.REUSE=='default': c.REUSE = tf.AUTO_REUSE
        with tf.variable_scope(c.scope, reuse=reuse):
            with tf.device(c._DEVICE):
                if c.SET_SEED_MANUALLY: tf.set_random_seed(c.SEED)
                out = tf.contrib.layers.flatten(inpt)
                for hidden in c.hiddens:
                    out = layers.fully_connected(out, num_outputs=hidden)
                    if c.DROPOUT: out = tf.nn.dropout(out, c.DROPOUT)
                    if layer_norm: out = layers.layer_norm(\
                            out, center=True, scale=True)
                out = layers.fully_connected(out, num_outputs=c.MAX_NUM_ACTIONS,\
                                activation_fn=c.FINAL_ACTIVATION)
                return out
                                                       
    c=config
    if not c.NETWORK: c.hiddens=[64]
    return lambda *args, **kwargs: _mlp_custom(c=c, *args, **kwargs)

# alias:
mlp = original_pathfinder_model

#def _decider_mlp(hiddens, inpt_list, num_actions, scope, seed_int, \
#                reuse=tf.AUTO_REUSE, layer_norm=False,config=None):
#    assert(len(hiddens[0])==len(inpt_list))
#    for nth_layer in range(len(hiddens)-1):
#        assert(len(hiddens[i+1]) <= len(hiddens[i]))
#        # TODO: update this for arbitrary models
#    with tf.variable_scope(scope, reuse=reuse):
#        with tf.device(config._DEVICE): # Require a config!
#            tf.set_random_seed(seed_int)
#            out = tf.contrib.layers.flatten(inpt)
#            for hidden in hiddens:
##                out = layers.fully_connected(out, num_outputs=hidden,\
##                        activation_fn=None)
##                out = tf.nn.relu(out)
#                out = layers.fully_connected(out, num_outputs=hidden)
#                out = tf.nn.dropout(out, 0.5)
#                if layer_norm:
#                    out = layers.layer_norm(out, center=True, scale=True)
##            q_out = layers.fully_connected(out, num_outputs=num_actions,\
##                        activation_fn=None)
##            q_out = tf.nn.tanh(q_out)
#            q_out = layers.fully_connected(out, num_outputs=num_actions,\
#                        activation_fn=tf.nn.tanh)
#            return q_out
#
#def decider_double_model(seed=None, config=None):
#    hiddens=[ (64,64), (32,) ]
##    if not 'scope' in *kwargs.keys(): 
##        print (kwargs)
##        kwargs['scope'] = str(seed
#    return lambda *args, **kwargs: _decider_mlp(hiddens, config=config, seed_int=int(seed), *args, **kwargs)
#
#
#def mlp(hiddens=[], layer_norm=False):
#    """This model takes as input an observation and returns values of all actions.
#
#    Parameters
#    ----------
#    hiddens: [int]
#        list of sizes of hidden layers
#
#    Returns
#    -------
#    q_func: function
#        q_function for DQN algorithm.
#    """
#    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)
#
#
#def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
#    with tf.variable_scope(scope, reuse=reuse):
#        out = inpt
#        with tf.variable_scope("convnet"):
#            for num_outputs, kernel_size, stride in convs:
#                out = layers.convolution2d(out,
#                                           num_outputs=num_outputs,
#                                           kernel_size=kernel_size,
#                                           stride=stride,
#                                           activation_fn=tf.nn.relu)
#        conv_out = layers.flatten(out)
#        with tf.variable_scope("action_value"):
#            action_out = conv_out
#            for hidden in hiddens:
#                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
#                if layer_norm:
#                    action_out = layers.layer_norm(action_out, center=True, scale=True)
#                action_out = tf.nn.relu(action_out)
#            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
#
#        if dueling:
#            with tf.variable_scope("state_value"):
#                state_out = conv_out
#                for hidden in hiddens:
#                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
#                    if layer_norm:
#                        state_out = layers.layer_norm(state_out, center=True, scale=True)
#                    state_out = tf.nn.relu(state_out)
#                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
#            action_scores_mean = tf.reduce_mean(action_scores, 1)
#            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
#            q_out = state_score + action_scores_centered
#        else:
#            q_out = action_scores
#        return q_out
#
#
#def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
#    """This model takes as input an observation and returns values of all actions.
#
#    Parameters
#    ----------
#    convs: [(int, int int)]
#        list of convolutional layers in form of
#        (num_outputs, kernel_size, stride)
#    hiddens: [int]
#        list of sizes of hidden layers
#    dueling: bool
#        if true double the output MLP to compute a baseline
#        for action scores
#
#    Returns
#    -------
#    q_func: function
#        q_function for DQN algorithm.
#    """
#
#    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)
#




# originally from templates.py:
#'''   Printing legend for condensed mode:
#    !  agent and goal       I  agent        -  immobile
#    @  goal and mobile      *  goal         o  mobile
#    <space>  empty          #  ERROR                        '''
#
#X,Y=0,1
## Warning: this function got all screwed up when copy-pasted.
#def print_state(start_state, mode, print_or_ret='print'):
#    S = ''
#    if type(start_state)==np.ndarray:
#        st = start_state
#    else:
#        st = start_state['state']
#        S += str(mode+':')
#    if mode=='matrices':
#        for i in range(st.shape[-1]):
#            S += str(st[:,:,i])
#    if mode=='condensed':
#        for y in range(st.shape[Y]):
#            for x in range(st.shape[X]):
#                if st[x,y,goalLayer] and st[x,y,agentLayer]: 
#                    S += str('!')
#                elif st[x,y,agentLayer]: 
#                    S += str('I')
#                elif st[x,y,goalLayer] and st[x,y,mobileLayer]:
#                    S += str('@')
#                elif st[x,y,goalLayer]: 
#                    S += str('*')
#                elif st[x,y,immobileLayer]: 
#                    S += str('-')
#                elif st[x,y,mobileLayer]: 
#                    S += str('o')
#                elif 0==np.sum(st[x,y,:]): 
#                    S += str(' ')
#                else: 
#                    S += str('#')
#                    print(S)
#                # raise Exception("Error")
#            S += str('\n')
#    if not type(start_state)==np.ndarray:
#        S += str("Flavor signal/goal id: ", start_state['flavor signal'])
#
#    if print_or_ret=='print': print(S)
#    else: return S
