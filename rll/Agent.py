import gym
import numpy as np


class Agent:
    def __init__(self, env: gym.Env):
        self.env = env

    def train_action(self, state):
        raise NotImplementedError

    def test_action(self, state):
        raise NotImplementedError

    def update(self, state, action, new_state, reward, done):
        # update inner knowledge
        raise NotImplementedError

    def episode(self, state):
        total_reward = 0
        done = False
        while not done:
            action = self.train_action(state)
            new_state, reward, done, info = self.env.step(action)
            total_reward += reward
            self.update(state, action, new_state, reward, done)
            state = new_state
        return total_reward

    def train(self,
              nb_episodes=1000,
              script_logger = None,
              average_window=100):
        all_rewards = []
        v = 0
        for i in range(nb_episodes):
            state = self.env.reset()
            reward = self.episode(state)
            all_rewards.append(reward)
            if script_logger:
                if len(all_rewards) < average_window + 1:
                    v = np.mean(all_rewards)
                else:
                    a = all_rewards[-average_window] if len(all_rewards) > average_window else 0
                    v += (reward - a)/average_window
                script_logger.log('Rewards', v)

    def test(self,
             nb_episodes=100,
             display=False):
        rewards = []
        for i in range(nb_episodes):
            episode_reward = 0
            state = self.env.reset()
            done = False
            while not done:
                if display:
                    self.env.render()
                a = self.test_action(state)
                state, step_reward, done, info = self.env.step(a)
                episode_reward += step_reward
            rewards.append(episode_reward)
        print('Rewards :', rewards)
