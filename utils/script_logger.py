import numpy as np


class ScriptLogger:
    def __init__(self, forward=None):
        # Send To is for variable monitoring
        self.forward = forward
        self.buffer = {}

    def log(self, var, value):
        if not var in self.buffer.keys():
            self.buffer[var] = []
        self.buffer[var].append(value)
        self.forward.log(var, value) #to say that we are updating var with value


    def read_buffer(self, var):
        if not var in self.buffer.keys():
            self.log('error', 'Attempted to access unknown key ({0} in Log'.format(var))
            var_buffer = []
        else:
            var_buffer = self.buffer[var]
            self.buffer[var] = []
        return var_buffer

    # standard logging
    def log_graphical(self, img):
        self.log('graphical', img)

    def log_error(self, err_string):
        self.log('error', err_string)

    def log_table_entry(self, dic):
        # the keys in dic should match for all entries
        self.log('table', dic)