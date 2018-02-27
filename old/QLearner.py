from old import DiscreteQMap, Grid
from phazcodelib.rll.QLearning import gymspace_to_grid
from phazcodelib.rll.Agent import Agent
import gym


class QLearner(Agent):
    def __init__(self,
                 env: gym.Env,
                 grid_resolution=10,
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

        state_grid: Grid = gymspace_to_grid(env.observation_space,
                                            resolution=grid_resolution)
        action_grid: Grid = gymspace_to_grid(env.action_space)
        self.Q = DiscreteQMap(state_grid, action_grid)
        super().__init__(env)

    def train_action(self, state):
        action = self.Q.probabilistic_action(state,
                        selection_mode=self.action_selection_mode,
                        epsilon=self.epsilon,
                        tau=self.tau)
        return action

    def test_action(self, state):
        action = self.Q.best_action(state)
        return action

    def update(self, state, action, new_state, reward, done):
        b_a_v = self.Q.best_action_value(new_state) if not done else 0
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] \
                                + self.alpha * (reward + self.gamma * b_a_v)
        self.epsilon *= 0.997
        return None

