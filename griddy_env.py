import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from griddy_render import *
from gym.envs.classic_control import rendering

class GriddyEnvOneHot(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: MultiDiscrete((4, 4), 4)
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move to the left
        1	Move to the right
        2	Move to the north
        3	Move to the south
    Reward:
        Reward is 0 for every step taken and 1 when goal is reached
    Starting State:
        Agent starts in random position and goal is always bottom right
    Episode Termination:
        Agent position is equal to goal position
        Solved Requirements
        Solved fast
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.n_squares_height = 4
        self.n_squares_width = 4

        self.keys_to_action = {
            ord('a'):0,
            ord('d'):1,
            ord('w'):2,
            ord('s'):3
        }

        self.OBJECT_TO_IDX = {
            'goal':1,
            'wall':2,
            'agent':3
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width))

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #state = np.array(self.state).reshape(4, 4)
        goal_pos = list(zip(*np.where(self.state[0] == 1)))[0]
        agent_pos = list(zip(*np.where(self.state[2] == 1)))[0]
        
        #move
        new_agent_pos = np.array(agent_pos)
        if action==0:
            new_agent_pos[1]-=1
        elif action==1:
            new_agent_pos[1]+=1
        elif action==2:
            new_agent_pos[0]-=1
        elif action==3:
            new_agent_pos[0]+=1    
        new_agent_pos = np.clip(new_agent_pos, 0, 3)
        
        self.state[2, agent_pos[0], agent_pos[1]] = 0 #moved from this position so it is empty
        self.state[2, new_agent_pos[0], new_agent_pos[1]] = 1 #moved to this position
        #self.state = tuple(self.state.flatten())
        
        #check if done
        done=False
        if np.all(np.array(goal_pos)==new_agent_pos):
            done=True
        
        #assign reward
        if not done:
            reward = 0
        elif self.steps_beyond_done is None:
            # Just arrived at the goal
            self.steps_beyond_done = 0
            reward = 1
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self, random_goal=False):
        state = np.full((len(self.OBJECT_TO_IDX), self.n_squares_height, self.n_squares_width), 0)
        if random_goal:
            agent_pos, goal_pos = np.random.choice(range(self.n_squares_height*self.n_squares_width), 2, replace=False)
            agent_pos, goal_pos = (agent_pos//self.n_squares_width, agent_pos%self.n_squares_width), (goal_pos//self.n_squares_width, goal_pos%self.n_squares_width)
            state[0, goal_pos[0], goal_pos[1]] = 1
        else:
            agent_pos = np.random.choice(range(self.n_squares_height*self.n_squares_width-1), 1, replace=False)[0]
            agent_pos = (agent_pos//self.n_squares_width, agent_pos%self.n_squares_width)
            state[0, self.n_squares_height-1, self.n_squares_width-1] = 1
            
        state[2, agent_pos[0], agent_pos[1]] = 1
        self.state = state
        self.steps_beyond_done = None
        return self.state 

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #horizontal grid lines
            for i in range(1, self.n_squares_height):
                track = rendering.Line((0,i*square_size_height), (screen_width,i*square_size_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #vertical grid lines
            for i in range(1, self.n_squares_width):
                track = rendering.Line((i*square_size_width, 0), (i*square_size_width, screen_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #the agent
            self.agent = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            #the goal
            self.goal = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.goal.set_color(0,255,0)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)
        if self.state is None: return
        goal_pos = list(zip(*np.where(self.state[0] == 1)))[0]
        agent_pos = list(zip(*np.where(self.state[2] == 1)))[0]

        agent_y = (self.n_squares_height-agent_pos[0]-0.5) * square_size_height
        agent_x = (agent_pos[1]+0.5) * square_size_width
        self.agenttrans.set_translation(agent_x, agent_y)

        goal_y = (self.n_squares_height-goal_pos[0]-0.5) * square_size_height
        goal_x = (goal_pos[1]+0.5) * square_size_width
        self.goaltrans.set_translation(goal_x, goal_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class GriddyEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: MultiDiscrete((4, 4), 4)
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move to the left
        1	Move to the right
        2	Move to the north
        3	Move to the south
    Reward:
        Reward is 0 for every step taken and 1 when goal is reached
    Starting State:
        Agent starts in random position and goal is always bottom right
    Episode Termination:
        Agent position is equal to goal position
        Solved Requirements
        Solved fast
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.n_squares_height = 4
        self.n_squares_width = 4

        self.keys_to_action = {
            ord('a'):0,
            ord('d'):1,
            ord('w'):2,
            ord('s'):3
        }

        self.OBJECT_TO_IDX = {
            'empty':0,
            'goal':1,
            'wall':2,
            'agent':3
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(np.full((2, self.n_squares_height, self.n_squares_width), len(self.OBJECT_TO_IDX)))

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #state = np.array(self.state).reshape(4, 4)
        goal_pos = list(zip(*np.where(self.state == 1)))[0]
        agent_pos = list(zip(*np.where(self.state == 3)))[0]
        
        #move
        new_agent_pos = np.array(agent_pos)
        if action==0:
            new_agent_pos[2]-=1
        elif action==1:
            new_agent_pos[2]+=1
        elif action==2:
            new_agent_pos[1]-=1
        elif action==3:
            new_agent_pos[1]+=1    
        new_agent_pos = np.clip(new_agent_pos, 0, 3)
        if self.state[new_agent_pos[0], new_agent_pos[1], new_agent_pos[2]]!=0:
            new_agent_pos[0]=1
        
        self.state[agent_pos[0], agent_pos[1], agent_pos[2]] = 0 #moved from this position so it is empty
        self.state[new_agent_pos[0], new_agent_pos[1], agent_pos[2]] = 3 #moved to this position
        #self.state = tuple(self.state.flatten())
        
        #check if done
        done=False
        if goal_pos==list(new_agent_pos):
            done=True
        
        #assign reward
        if not done:
            reward = 0
        elif self.steps_beyond_done is None:
            # Just arrived at the goal
            self.steps_beyond_done = 0
            reward = 1
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self, random_goal=False):
        state = [0]*self.n_squares_height*self.n_squares_width
        if random_goal:
            agent_pos, goal_pos = np.random.choice(range(self.n_squares_height*self.n_squares_width), 2, replace=False)
            state[goal_pos] = 1
        else:
            agent_pos = np.random.choice(range(self.n_squares_height*self.n_squares_width-1), 1, replace=False)[0]
            state[15] = 1
        state[agent_pos] = 3
        state.extend([self.n_squares_height*self.n_squares_width*2])
        self.state = np.array(state).reshape(3, self.n_squares_height, self.n_squares_width)
        self.steps_beyond_done = None
        return self.state 

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #horizontal grid lines
            for i in range(1, self.n_squares_height):
                track = rendering.Line((0,i*square_size_height), (screen_width,i*square_size_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #vertical grid lines
            for i in range(1, self.n_squares_width):
                track = rendering.Line((i*square_size_width, 0), (i*square_size_width, screen_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #the agent
            self.agent = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            #the goal
            self.goal = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.goal.set_color(0,255,0)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)
        if self.state is None: return
        goal_pos = list(zip(*np.where(self.state == 1)))[0]
        agent_pos = list(zip(*np.where(self.state == 3)))[0]

        agent_y = (self.n_squares_height-agent_pos[0]-0.5) * square_size_height
        agent_x = (agent_pos[1]+0.5) * square_size_width
        self.agenttrans.set_translation(agent_x, agent_y)

        goal_y = (self.n_squares_height-goal_pos[0]-0.5) * square_size_height
        goal_x = (goal_pos[1]+0.5) * square_size_width
        self.goaltrans.set_translation(goal_x, goal_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class GriddyEnvBackup(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: MultiDiscrete((4, 4), 4)
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move to the left
        1	Move to the right
        2	Move to the north
        3	Move to the south
    Reward:
        Reward is 0 for every step taken and 1 when goal is reached
    Starting State:
        Agent starts in random position and goal is always bottom right
    Episode Termination:
        Agent position is equal to goal position
        Solved Requirements
        Solved fast
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.n_squares_height = 4
        self.n_squares_width = 4

        self.keys_to_action = {
            ord('a'):0,
            ord('d'):1,
            ord('w'):2,
            ord('s'):3
        }

        self.OBJECT_TO_IDX = {
            'empty':0,
            'goal':1,
            'wall':2,
            'agent':3
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(np.full((self.n_squares_height, self.n_squares_width), len(self.OBJECT_TO_IDX)))

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #state = np.array(self.state).reshape(4, 4)
        goal_pos = list(zip(*np.where(self.state == 1)))[0]
        agent_pos = list(zip(*np.where(self.state == 3)))[0]
        
        #move
        new_agent_pos = np.array(agent_pos)
        if action==0:
            new_agent_pos[1]-=1
        elif action==1:
            new_agent_pos[1]+=1
        elif action==2:
            new_agent_pos[0]-=1
        elif action==3:
            new_agent_pos[0]-=1    
        new_agent_pos = np.clip(new_agent_pos, 0, 3)
        
        self.state[agent_pos[0], agent_pos[1]] = 0 #moved from this position so it is empty
        self.state[new_agent_pos[0], new_agent_pos[1]] = 3 #moved to this position
        #self.state = tuple(self.state.flatten())
        
        #check if done
        done=False
        if goal_pos==list(new_agent_pos):
            done=True
        
        #assign reward
        if not done:
            reward = 0
        elif self.steps_beyond_done is None:
            # Just arrived at the goal
            self.steps_beyond_done = 0
            reward = 1
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self, random_goal=False):
        state = [0]*16
        if random_goal:
            agent_pos, goal_pos = np.random.choice(range(16), 2, replace=False)
            state[goal_pos] = 1
        else:
            agent_pos = np.random.choice(range(15), 1, replace=False)[0]
            state[15] = 1
        state[agent_pos] = 3
        self.state = np.array(state).reshape(4, 4)
        self.steps_beyond_done = None
        return self.state 

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #horizontal grid lines
            for i in range(1, self.n_squares_height):
                track = rendering.Line((0,i*square_size_height), (screen_width,i*square_size_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #vertical grid lines
            for i in range(1, self.n_squares_width):
                track = rendering.Line((i*square_size_width, 0), (i*square_size_width, screen_height))
                track.set_color(0,0,0)
                self.viewer.add_geom(track)
            #the agent
            self.agent = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            #the goal
            self.goal = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.goal.set_color(0,255,0)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)
        if self.state is None: return
        goal_pos = list(zip(*np.where(self.state == 1)))[0]
        agent_pos = list(zip(*np.where(self.state == 3)))[0]

        agent_y = (self.n_squares_height-agent_pos[0]-0.5) * square_size_height
        agent_x = (agent_pos[1]+0.5) * square_size_width
        self.agenttrans.set_translation(agent_x, agent_y)

        goal_y = (self.n_squares_height-goal_pos[0]-0.5) * square_size_height
        goal_x = (goal_pos[1]+0.5) * square_size_width
        self.goaltrans.set_translation(goal_x, goal_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
