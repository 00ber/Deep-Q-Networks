import os
import torch
from pathlib import Path

from agent import DQNAgent, MetricLogger
from wrappers import make_starpilot
import os
from train import train, fill_memory


env = make_starpilot()

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

checkpoint = None 
# checkpoint = Path('checkpoints/latest/airstriker_net_3.chkpt')

path = "checkpoints/procgen-starpilot-dqn"
save_dir = Path(path) 

isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

logger = MetricLogger(save_dir)

print("Training Vanilla DQN Agent!")
agent = DQNAgent(
    state_dim=(1, 64, 64), 
    action_dim=env.action_space.n,
    save_dir=save_dir, 
    batch_size=128,
    checkpoint=checkpoint,  
    exploration_rate_decay=0.9995,
    exploration_rate_min=0.05,
    training_frequency=1, 
    target_network_sync_frequency=500,
    max_memory_size=50000,
    learning_rate=0.0005,

)

fill_memory(agent, env, 300)
train(agent, env, logger)
