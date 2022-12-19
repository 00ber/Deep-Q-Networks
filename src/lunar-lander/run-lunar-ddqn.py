import os
import torch
from pathlib import Path

from agent import DDQNAgent, DDQNAgentWithStepDecay, MetricLogger
from wrappers import make_lunar
import os
from train import train, fill_memory
from params import hyperparams

env = make_lunar()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

checkpoint = None 
# checkpoint = Path('checkpoints/latest/airstriker_net_3.chkpt')

path = "checkpoints/lunar-lander-ddqn-rc"
save_dir = Path(path) 

isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

logger = MetricLogger(save_dir)

print("Training DDQN Agent!")
agent = DDQNAgentWithStepDecay(
    state_dim=8, 
    action_dim=env.action_space.n,
    save_dir=save_dir, 
    checkpoint=checkpoint,  
    **hyperparams
)
# agent = DDQNAgent(
#     state_dim=8, 
#     action_dim=env.action_space.n,
#     save_dir=save_dir, 
#     checkpoint=checkpoint,  
#     **hyperparams
# )

# fill_memory(agent, env, 5000)
train(agent, env, logger)
