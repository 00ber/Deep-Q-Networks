import random, datetime
from pathlib import Path
import retro as gym
from collections import namedtuple, deque
from itertools import count

import torch
import matplotlib
import matplotlib.pyplot as plt
from agent import MyAgent, MyDQN, MetricLogger
from wrappers import make_env


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# env = gym.make(game='Airstriker-Genesis')
# env = AirstrikerDiscretizer(env)
# env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env)
# if gym.__version__ < '0.26':
#     env = FrameStack(env, num_stack=4, new_step_api=True)
# else:
#     env = FrameStack(env, num_stack=4)

env = make_env()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

agent = MyAgent(state_dim=(1, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 100000
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
        
        # Check if end of game
        if done or info["gameover"] == 1:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
