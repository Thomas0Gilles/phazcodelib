from old import QLearner


class SarsaLearner(QLearner):
    def update(self, state, action, new_state, reward, done):
        new_action = self.train_action(new_state)
        d = reward + self.gamma*self.Q[new_state, new_action] - self.Q[state, action]
        self.Q[state, action] += self.alpha * d
        return new_action

    def episode(self, state):
        total_reward = 0
        action = self.train_action(state)
        done = False
        while not done:
            new_state, reward, done, info = self.env.step(action)
            total_reward += reward
            action = self.update(state, action, new_state, reward, done)
            state = new_state
        return total_reward
