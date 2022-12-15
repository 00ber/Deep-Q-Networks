import retro as gym
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = gym.make(game='Airstriker-Genesis')
env.reset()


print(device)
def main():
    done = False 
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        print(reward)
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()
