import datetime
from pathlib import Path
from agent import DQNAgent, DDQNAgent, MetricLogger
from wrappers import make_lunar


env = make_lunar()

env.reset()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# checkpoint = Path('checkpoints/lunar-lander-dueling-ddqn/airstriker_net_2.chkpt')
checkpoint = Path('checkpoints/lunar-lander-dqn-rc/airstriker_net_1.chkpt')

logger = MetricLogger(save_dir)

print("Testing Double DQN Agent!")
agent = DDQNAgent(
    state_dim=8, 
    action_dim=env.action_space.n,
    save_dir=save_dir, 
    batch_size=512,
    checkpoint=checkpoint,  
    exploration_rate_decay=0.999995,
    exploration_rate_min=0.05,
    training_frequency=1, 
    target_network_sync_frequency=200,
    max_memory_size=50000,
    learning_rate=0.0005,
    load_replay_buffer=False

)
agent.exploration_rate = agent.exploration_rate_min

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        # agent.cache(state, next_state, action, reward, done)

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
