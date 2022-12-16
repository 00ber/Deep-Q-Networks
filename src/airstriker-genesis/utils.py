import gym 
import numpy as np


# Airstrikerラッパー
class AirstrikerDiscretizer(gym.ActionWrapper):
    # 初期化
    def __init__(self, env):
        super(AirstrikerDiscretizer, self).__init__(env)
        buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
        actions = [['LEFT'], ['RIGHT'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    # 行動の取得
    def action(self, a):
        return self._actions[a].copy()
