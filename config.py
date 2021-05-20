import numpy as np
from networkx import grid_graph, to_numpy_matrix

config = {
    'adj_mat' : to_numpy_matrix(grid_graph(dim=(4,4)), dtype=np.bool),
    'initial_fire' : np.asarray([False, False, True]+[False]*13, dtype=np.bool),
    'burn_prob': 0.5,
    'n_defend': 2,
}