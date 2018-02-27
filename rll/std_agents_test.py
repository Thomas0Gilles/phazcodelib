from phazcodelib.utils import ScriptLogger, VariableEvolutionDisplay
from .std_agents_config import *

slg = ScriptLogger(forward=VariableEvolutionDisplay())

QL.train(nb_episodes=1000, script_logger=slg)
QL.test(nb_episodes=5, display=True)


