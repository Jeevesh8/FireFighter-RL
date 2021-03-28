"""Fire Fighting on Graphs reinforcement learning environment."""

import dm_env
from dm_env import specs
import numpy as np
import networkx as nx


class FireFighter(dm_env.Environment):
    """A FireFighter environment built on the `dm_env.Environment` class.

    The agent must choose which vertices to defend at each time step.

    The observation is a 3-tuple the first element is a
    boolean adjacency matrix of shape (|V|, |V|). The 2nd & 3rd are also boolean arrays,
    but with shape (|V|,) corres. to whether a vertex is burnt or defended respectively.

    The actions are discrete, and must be a (|V|,) sized boolean vector indicating
    which vertices to defend.

    The episode terminates when no more vertices can be burnt.
    """

    def __init__(
        self,
        adj_mat: np.ndarray,
        initial_fire: np.ndarray,
        burn_prob: float = 0.5,
        seed=42,
    ):
        """Initializes a new Catch environment.

        Args:
          adj_mat: Boolean np.ndarray representing the adjacency matrix
          seed: random seed for the RNG.
        """
        self.adj_mat = adj_mat
        self.inital_fire = initial_fire
        self.burn_prob = burn_prob
        self._rng = np.random.RandomState(seed)

        self._reset_next_step = True

    def reset(self):
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False

        self.burned = self.inital_fire.copy()
        self.defended = np.array([False] * self.adj_mat.shape[0], dtype=np.bool)

        self.graph = nx.convert_matrix.from_numpy_matrix(self.adj_mat)
        self.nodes = self.graph.nodes()
        return dm_env.restart(self._observation())

    def burn_vertices(self):
        """
        Burns any vertex neighboring to a vertex on fire, and not defended/previously burnt,
        with probability self.burn_prob
        """
        burnable = np.logical_and(
            np.any(self.adj_mat[self.burned], axis=0),
            np.logical_not(np.logical_or(self.defended, self.burned)),
        )

        self._reset_next_step = not np.any(burnable)

        to_burn = np.random.uniform(size=burnable.shape) < self.burn_prob

        self.burned[np.logical_and(burnable, to_burn)] = True

    def step(self, action: np.ndarray):
        """Updates the environment according to the action."""
        if self._reset_next_step:
            return self.reset()

        self.defended = np.logical_or(self.defended, action)

        self.burn_vertices()

        if self._reset_next_step:
            return dm_env.termination(reward=0.0, observation=self._observation())

        return dm_env.transition(reward=-1.0, observation=self._observation())

    def observation_spec(self):
        """Returns the observation spec."""
        return (
            specs.Array(
                shape=self.adj_mat.shape, dtype=np.bool, name="adjacency_matrix"
            ),
            specs.Array(
                shape=(self.adj_mat.shape[0],), dtype=np.bool, name="burned"
            ),
            specs.Array(
                shape=(self.adj_mat.shape[0],), dtype=np.bool, name="defended"
            ),
        )

    def action_spec(self):
        """Returns the action spec."""
        return specs.Array(
            shape=(self.adj_mat.shape[0],), dtype=np.bool, name="defend_vertices"
        )

    def _observation(self):
        return (self.adj_mat.copy(), self.burned.copy(), self.defended.copy())

    def set_state(self, state):
        self.burned, self.defended = state