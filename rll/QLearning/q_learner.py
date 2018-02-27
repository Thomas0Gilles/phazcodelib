from phazcodelib.rll.QLearning import SpaceGrid
from phazcodelib.rll.Agent import Agent
from phazcodelib.utils import ndargmax
import gym
import numpy as np


class QLearner(Agent):
    def __init__(self,
                 env: gym.Env,
                 boxes_resolution=10,
                 alpha=0.5,
                 gamma=1,
                 action_selection_mode='epsilon-greedy',
                 epsilon=0.1,
                 tau=1):
        self.alpha = alpha
        self.gamma = gamma
        self.action_selection_mode = action_selection_mode
        self.epsilon = epsilon
        self.tau = tau

        self.Q = SpaceGrid(env=env, boxes_resolution=boxes_resolution)
        super().__init__(env)

    def act(self, state, explore=True):
        best_action = ndargmax(self.Q[state])
        if explore and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # random !
        return best_action

    def update(self):
        for i in range(len(self.log) - 1):
            state, action = self.log[i]['state'], self.log[i]['action']
            new_state, reward = self.log[i+1]['state'], self.log[i]['reward']
            self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
                + self.alpha * (reward + self.gamma * np.argmax(self.Q[state]))
        self.log = []
        self.epsilon *= 0.997
        return None


class SarsaLearner(QLearner):
    def update(self):
        for i in range(len(self.log)-1):
            reward = self.log[i]['reward']
            state, action = self.log[i]['state'], self.log[i]['action']
            new_state, new_action = self.log[i+1]['state'], self.log[i+1]['action']
            d = reward + self.gamma*self.Q[new_state, new_action] - self.Q[state, action]
            self.Q[state, action] += self.alpha * d
        return None
