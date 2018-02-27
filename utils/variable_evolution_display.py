import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import numpy as np

class VariableEvolutionDisplay:
    def __init__(self, update_frequency=1):
        self.update_period = 1 / update_frequency
        self.last_update_time = {}
        self.values = {}
        self.ax = {}

    def _init_var_graph(self, var):
        self.last_update_time[var] = time.time()
        self.values[var] = []
        fig = plt.figure()
        self.ax[var] = fig.add_subplot(1, 1, 1)
        plt.title('Evolution of Variable ' + var)
        self.ax[var].set_xlim(auto=True)
        self.ax[var].set_ylim(auto=True)
        plt.draw()

    def _update_var_value(self, var, value):
        self.values[var].append(value)

    def log(self, var, value):
        if var not in self.last_update_time:
            self._init_var_graph(var)
            self._update_var_value(var, value)
        if time.time() > self.last_update_time[var] + self.update_period:
            self.last_update_time[var] = time.time()
            self._update_var_graph(var)

    def _update_var_graph(self, var):
        ax = self.ax[var]
        values = self.values[var]
        ax.clear()
        size = len(values)
        if size < 200:
            ax.plot(range(size), values)
        else:
            i_range = range(0, size, int(size/200))
            ax.plot(i_range, np.take(values, i_range))
        plt.draw()
        plt.pause(1e-3)

class AverageVariableEvolutionDisplay(VariableEvolutionDisplay):
    # for plotting averaging !
    def __init__(self, average_window, **kwargs):
        self.average_window = average_window
        super().__init__(**kwargs)

    def _update_var_value(self, var, value):
        last_value, size = self.values[var][-1], len(self.values[var])
        if size < self.average_window:
            new_value = (last_value*size+value)/(size+1)
        else:
            new_value = last_value + (value - self.values[var][-self.average_window])/self.average_window
        self.values[var].append(new_value)
