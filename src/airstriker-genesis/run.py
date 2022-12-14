from keras.models import Sequential
from keras.layers import Dense
import retro as gym


class DQNAgent:
    def create_model(self):
        model = Sequential()

buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

    
env = gym.make(game='Airstriker-Genesis')
env.reset()

done = False


# DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
# discrete_obs_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

# print(discrete_obs_win_size)

while not done:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        obs = env.reset()
env.close()
    




