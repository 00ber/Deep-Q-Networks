import os
import random, datetime
from pathlib import Path
import retro as gym
from collections import namedtuple, deque
from itertools import count

import torch
import matplotlib
import matplotlib.pyplot as plt
# from agent import MyAgent, MyDQN, MetricLogger
from cartpole import MyAgent, MetricLogger
from wrappers import make_env
import pickle
import gym 

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# env = make_env()
env = gym.make('CartPole-v1')

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

path = "checkpoints/cartpole/latest"
save_dir = Path(path) 

isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

# save_dir.mkdir(parents=True)


checkpoint = None 
# checkpoint = Path('checkpoints/latest/airstriker_net_3.chkpt')

# For cartpole
n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)
agent = MyAgent(state_dim=n_observations, action_dim=n_actions, save_dir=save_dir, checkpoint=checkpoint, reset_exploration_rate=True)

# For airstriker
# agent = MyAgent(state_dim=(1, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint, reset_exploration_rate=True)


logger = MetricLogger(save_dir)

episodes = 10000000
for e in range(episodes):

    state = env.reset()
    # Play the game!
    while True:
       
        # print(state.shape)
        # Run agent on the state
        action = agent.act(state)
        
        # Agent performs action
        next_state, reward, done, info = env.step(action)
        
        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = agent.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state
        
        # # Check if end of game (for airstriker)
        # if done or info["gameover"] == 1:
        #     break
        # Check if end of game (for cartpole)
        if done:
            break

    logger.log_episode(e)

    if e % 20 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
