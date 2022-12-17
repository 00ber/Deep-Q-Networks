import datetime
from pathlib import Path
from itertools import count
from agent import MyAgent,  MetricLogger
from wrappers import make_env


env = make_env()

env.reset()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2022-12-16T01-20-33/airstriker__net_1.chkpt')
agent = MyAgent(state_dim=(1, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
agent.exploration_rate = agent.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()

    while True:

        env.render()

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        agent.cache(state, next_state, action, reward, done)

        logger.log_step(reward, None, None)

        state = next_state

        if done or info["gameover"] == 1:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )
