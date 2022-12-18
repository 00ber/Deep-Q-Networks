import datetime
from pathlib import Path
from itertools import count
from agent import DQNAgent,  MetricLogger
from wrappers import make_env, make_starpilot


env = make_starpilot()

env.reset()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/procgen-starpilot-dqn/airstriker_net_3.chkpt')

agent = DQNAgent(
    state_dim=(1, 64, 64), 
    action_dim=env.action_space.n,
    save_dir=save_dir, 
    batch_size=256,
    checkpoint=checkpoint, 
    reset_exploration_rate=True, 
    exploration_rate_decay=0.999999,
    training_frequency=10, 
    target_network_sync_frequency=200,
    max_memory_size=3000,
    learning_rate=0.001,
    save_frequency=2000

)
agent.exploration_rate = agent.exploration_rate_min

# logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        agent.cache(state, next_state, action, reward, done)

        # logger.log_step(reward, None, None)

        state = next_state

        if done:
            break

    # logger.log_episode()

    # if e % 20 == 0:
    #     logger.record(
    #         episode=e,
    #         epsilon=agent.exploration_rate,
    #         step=agent.curr_step
    #     )
