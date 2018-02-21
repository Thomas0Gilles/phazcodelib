import numpy as np
from phazcodelib.rll.QLearning import Grid

def tpl(l):
    if isinstance(l, tuple) or isinstance(l, list):
        return tuple(l)
    elif isinstance(l, int) or isinstance(l, bool):
        return l,
    else:
        raise Exception('Error')

class DiscreteQMap: # for Q(s, a) s state a action

    def __init__(self,
                 state_grid: Grid,
                 action_grid: Grid):
        self.action_grid = action_grid
        self.state_grid = state_grid
        self.Q = np.zeros(self.state_grid.shape + self.action_grid.shape)

    def _coords(self, state, action):
        s = self.state_grid[state]
        a = self.action_grid[action]
        return tpl(s)+tpl(a)

    def __getitem__(self, key):
        state, action = key
        return self.Q[self._coords(state, action)]

    def __setitem__(self, key, value):
        state, action = key
        self.Q[self._coords(state, action)] = value


    def action_values(self, state):
        return self.Q[self.state_grid[state]]

    def best_action(self, state):
        am = np.argmax(self.action_values(state))
        action = self.action_grid[am]
        return action

    def best_action_value(self, state):
        return np.max(self.action_values(state))

    def probabilistic_action(self, state,
                             selection_mode = 'epsilon-greedy',
                             epsilon=0.1,
                             tau=1):
        if selection_mode == 'epsilon-greedy':
            if np.random.rand() < epsilon:
                action = self.action_grid.sample()
            else:
                action = self.best_action(state)
        elif selection_mode == 'softmax':
            e = np.exp(self.action_values(state).flatten() / tau)
            c = np.random.choice(range(len(e)), p=e / np.sum(e))
            action = self.action_grid[c]
        return action



