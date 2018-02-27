import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.tuple_space import Tuple
from gym.spaces.box import Box

class OneDimGrid:
    def __init__(self, size):
        self.nb_dims = 1
        self.shape = (size,)
        self.size = size

    def sample(self):
        return np.random.randint(self.size)

    def __getitem__(self, i):
        assert isinstance(i, bool) or 0 <= i < self.size
        return int(i)

    def __repr__(self):
        return '1-dimensional Grid of size '+str(self.size)

class Grid:
    '''
    A grid
    corresponds to the cartesian space [0..(n1-1)]x[0..(n2-1)]x..
    '''
    def __init__(self, shape: tuple):
        self.shape = shape
        self.nb_dims = len(self.shape)
        self.size = np.product(self.shape)

    def sample(self):
        return tuple([np.random.randint(m) for m in self.shape])

    def __getitem__(self, ii):
        if isinstance(ii, int) or isinstance(ii, float):
            # try to get a multi-index
            m_i = np.unravel_index(ii, self.shape)
            return m_i
        elif ((isinstance(ii, tuple) or isinstance(ii, list))
                and len(ii) == self.nb_dims):
            return tuple([int(i) for i in ii])
        elif len(ii) != self.nb_dims:
            raise Exception('Grid shape and key mismatch ({O}!={1}'.format(ii, self.shape))
        else:
            raise Exception('Unknown Error')

    def __repr__(self):
        return '{0}-dimensional Grid with sizes {1}'.format(self.nb_dims, self.shape)

class QuantizeGrid(Grid):
    # quantize a space
    def __init__(self, low, high, resolution):
        self.low = low
        self.high = high
        self.resolution = resolution
        super().__init__(resolution)

    def __getitem__(self, point):
        if isinstance(point, int) or isinstance(point[0], int):
            # Access via int values
            return super().__getitem__(point)
        else:
            range = self.high - self.low
            float_coords = self.resolution * (point - self.low) / range
            int_coords = [int(x) for x in float_coords]
            return super().__getitem__(int_coords)


# GYM specific
def gymspace_to_grid(space, resolution = None):
    if isinstance(space, Discrete):
        return OneDimGrid(space.n)
    elif isinstance(space, Tuple):
        # tuple of Discrete
        return Grid([s.n for s in space.spaces])
    elif isinstance(space, Box):
        assert resolution is not None
        r = space.shape[0]*(resolution,)
        return QuantizeGrid(low=space.low,
                            high=space.high,
                            resolution=r)
    else:
        raise Exception('Unknown space class')

