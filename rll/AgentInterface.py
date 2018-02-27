import gym
import numpy as np


class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.exploration = True
        self.log = []  # episode log

    def act(self, state, explore):
        raise NotImplementedError

    def step_update(self, state, action, new_state, reward):
        # updates knowledge each time an action is taken
        pass

    def episode_update(self):
        # update inner knowledge based on log
        pass

    def episode(self, state, explore=True, display=False):
        total_reward = 0
        done = False
        while not done:
            action = self.act(state, explore)
            new_state, reward, done, info = self.env.step(action)
            self.step_update(state, action, new_state, reward)
            self.log.append(dict(state=state, action=action, reward=reward))
            state = new_state
            total_reward += reward
            if display:
                self.env.render()
        return total_reward

    def train(self, nb_episodes=1000, script_logger=None):
        for i in range(nb_episodes):
            reward = self.episode(self.env.reset())
            self.episode_update()
            if script_logger:
                script_logger.log('Rewards', reward)


    def test(self, nb_episodes=100, display=False):
        rewards = [self.episode(self.env.reset(), explore=False, display=display) for _ in range(nb_episodes)]
        print('All Rewards :', rewards)
        return rewards
