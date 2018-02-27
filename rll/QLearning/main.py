import gym
from phazcodelib.rll.QLearning.q_learner import QLearner
from phazcodelib.utils import ScriptLogger, VariableEvolutionDisplay

env = gym.make('MountainCar-v0')

QL = QLearner(env,
              boxes_resolution=20,
              gamma=0.99,
              alpha=1 / 10,
              tau=20,
              action_selection_mode='epsilon-greedy',
              epsilon=1 / 100)

slg = ScriptLogger(forward=VariableEvolutionDisplay())

QL.train(nb_episodes=1000, script_logger=slg)
QL.test(nb_episodes=5, display=True)
