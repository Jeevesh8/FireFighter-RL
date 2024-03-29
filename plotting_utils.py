import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def visualize_graph(adj_mat, burned, defended):
    g = nx.convert_matrix.from_numpy_matrix(adj_mat)
    node_color = np.where(burned, 1.0, 0.2)
    node_color = np.where(defended, 0.6, node_color)
    nx.draw_spectral(g, node_color=node_color.tolist())
    plt.plot()
    plt.show()
