import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import numpy as np

class GraphicalVariableEvolution:
    def __init__(self, update_frequency=1):
        self.update_period = 1 / update_frequency
        self.last_update_time = {}
        self.values = {}
        self.ax = {}

    def update_warning(self, var, value):
        if not var in self.last_update_time:
            self.last_update_time[var] = time.time()
            self.values[var] = []
            fig = plt.figure()
            self.ax[var] = fig.add_subplot(1, 1, 1)
            plt.title('Evolution of Variable '+var)
            self.ax[var].set_xlim(auto=True)
            self.ax[var].set_ylim(auto=True)
            plt.draw()
        self.values[var].append(value)
        if time.time() > self.last_update_time[var] + self.update_period:
            self.last_update_time[var] = time.time()
            self.update_variable_graph(var)

    def update_variable_graph(self, var):
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

