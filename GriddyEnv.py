import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from griddy_render import *
from gym.envs.classic_control import rendering

class GriddyEnv(gym.Env):
    """
    Description:
        A grid world where you have to reach the goal
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

    def __init__(self, width=4, height=4):
        self.n_squares_height = width
        self.n_squares_width = height

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

    def reset(self, random_goal=False):
        self.n_steps=0
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
        return np.copy(self.state )

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.n_steps+=1
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
        new_agent_pos[0] = np.clip(new_agent_pos[0], 0, self.n_squares_height-1)
        new_agent_pos[1] = np.clip(new_agent_pos[1], 0, self.n_squares_width-1)
        
        self.state[2, agent_pos[0], agent_pos[1]] = 0 #moved from this position so it is empty
        self.state[2, new_agent_pos[0], new_agent_pos[1]] = 1 #moved to this position
        
        #check if done
        done=False
        if np.all(np.array(goal_pos)==new_agent_pos):
            done=True
            
        #assign reward
        if not done:
            reward = 0
            if self.n_steps>=1000:
                self.steps_beyond_done = 0
                done=True
        elif self.steps_beyond_done is None:
            # Just arrived at the goal
            self.steps_beyond_done = 0
            reward = 1
        else:
            if self.steps_beyond_done >= 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.copy(self.state), reward, done, {}

    def render(self, values=None, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #add invisible squares for visualising state values
            l, r, t, b = -square_size_width/2, square_size_width/2, square_size_height/2, -square_size_height/2
            self.squares = [[rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            sq_transforms = [[rendering.Transform() for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            for i in range(0, self.n_squares_height):
                for j in range(0, self.n_squares_width):
                    self.squares[i][j].add_attr(sq_transforms[i][j])
                    self.viewer.add_geom(self.squares[i][j])
                    sq_x, sq_y = self.convert_pos_to_xy((i, j), (square_size_width, square_size_height))
                    sq_transforms[i][j].set_translation(sq_x, sq_y)
                    self.squares[i][j].set_color(1, 1, 1)
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
            #self.agent = rendering.Image('robo.jpg', width=square_size_width/2, height=square_size_height/2)
            l, r, t, b = -square_size_width/4, square_size_width/4, square_size_height/4, -square_size_height/4
            self.agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            #self.agent = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            #the goal
            self.goal = make_oval(width=square_size_width/4, height=square_size_height/4)
            self.goal.set_color(1,0,1)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)
        if self.state is None: return
        goal_pos = list(zip(*np.where(self.state[0] == 1)))[0]
        agent_pos = list(zip(*np.where(self.state[2] == 1)))[0]

        agent_x, agent_y = self.convert_pos_to_xy(agent_pos, (square_size_width, square_size_height))
        self.agenttrans.set_translation(agent_x, agent_y)

        goal_x, goal_y = self.convert_pos_to_xy(goal_pos, (square_size_width, square_size_height))
        self.goaltrans.set_translation(goal_x, goal_y)
        if values is not None:
            maxval, minval = values.max(), values.min()
            rng = maxval-minval
            for i, row in enumerate(values):
                for j, val in enumerate(row):
                    if rng==0: col=1
                    else: col=(maxval-val)/rng
                    self.squares[i][j].set_color(col, 1, col)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def convert_pos_to_xy(self, pos, size):
        x = (pos[1]+0.5) * size[0]
        y = (self.n_squares_height-pos[0]-0.5) * size[1]
        return x, y

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

'''values = np.array([[0.73509189, 0.77378094, 0.81450625, 0.857375  ],
       [0.77378094, 0.81450625, 0.857375  , 0.9025    ],
       [0.81450625, 0.857375  , 0.9025    , 0.95      ],
       [0.857375  , 0.9025    , 0.95      , 0        ]])
values = np.array([[0, 0, 0, 0  ],
       [0, 0, 0  , 0    ],
       [0, 0  , 0    , 0      ],
       [0, 0    , 0      , 0        ]])
env=GriddyEnv()
env.reset()
env.render(values)'''

class GriddyEnvAnton(gym.Env):
#     test change
    """
    Description:
        A grid world where you have to reach the goal
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

    def __init__(self, width=4, height=4):
        self.n_squares_height = width
        self.n_squares_width = height

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
        return np.copy(self.state )

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
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
        new_agent_pos[0] = np.clip(new_agent_pos[0], 0, self.n_squares_height-1)
        new_agent_pos[1] = np.clip(new_agent_pos[1], 0, self.n_squares_width-1)
        
        self.state[2, agent_pos[0], agent_pos[1]] = 0 #moved from this position so it is empty
        self.state[2, new_agent_pos[0], new_agent_pos[1]] = 1 #moved to this position
        
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
            if self.steps_beyond_done >= 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.copy(self.state), reward, done, {}

    def render(self, values=None, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height/self.n_squares_height
        square_size_width = screen_width/self.n_squares_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #add invisible squares for visualising state values
            l, r, t, b = -square_size_width/2, square_size_width/2, square_size_height/2, -square_size_height/2
            self.squares = [[rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)]) for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            sq_transforms = [[rendering.Transform() for j in range(0, self.n_squares_width)] for i in range(0, self.n_squares_height)]
            for i in range(0, self.n_squares_height):
                for j in range(0, self.n_squares_width):
                    self.squares[i][j].add_attr(sq_transforms[i][j])
                    self.viewer.add_geom(self.squares[i][j])
                    sq_x, sq_y = self.convert_pos_to_xy((i, j), (square_size_width, square_size_height))
                    sq_transforms[i][j].set_translation(sq_x, sq_y)
                    self.squares[i][j].set_color(1, 1, 1)
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
            #self.agent = rendering.Image('robo.jpg', width=square_size_width/2, height=square_size_height/2)
            l, r, t, b = -square_size_width/4, square_size_width/4, square_size_height/4, -square_size_height/4
            self.agent = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            #self.agent = make_oval(width=square_size_width/2, height=square_size_height/2)
            self.agenttrans = rendering.Transform()
            self.agent.add_attr(self.agenttrans)
            self.viewer.add_geom(self.agent)
            #the goal
            self.goal = make_oval(width=square_size_width/4, height=square_size_height/4)
            self.goal.set_color(1,0,1)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)
        if self.state is None: return
        goal_pos = list(zip(*np.where(self.state[0] == 1)))[0]
        agent_pos = list(zip(*np.where(self.state[2] == 1)))[0]

        agent_x, agent_y = self.convert_pos_to_xy(agent_pos, (square_size_width, square_size_height))
        self.agenttrans.set_translation(agent_x, agent_y)

        goal_x, goal_y = self.convert_pos_to_xy(goal_pos, (square_size_width, square_size_height))
        self.goaltrans.set_translation(goal_x, goal_y)
        if values is not None:
            maxval, minval = values.max(), values.min()
            rng = maxval-minval
            for i, row in enumerate(values):
                for j, val in enumerate(row):
                    if rng==0: col=1
                    else: col=(maxval-val)/rng
                    self.squares[i][j].set_color(col, 1, col)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def convert_pos_to_xy(self, pos, size):
        x = (pos[1]+0.5) * size[0]
        y = (self.n_squares_height-pos[0]-0.5) * size[1]
        return x, y

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

