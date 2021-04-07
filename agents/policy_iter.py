import random
from typing import List

import numpy as np

from agents.utils import numpy_dict

class policy_iter_agent():
    def __init__(self, n_defend):
        self.n_defend = n_defend
        self.policy = numpy_dict()
    
    def _get_all_actions(self, defendable, n_defend)->List[np.ndarray]:
        if n_defend<=0 or n_defend>=np.sum(defendable):
            return [np.copy(defendable)]
        
        for (i,elem) in enumerate(defendable):
            if elem:
                return ([np.concatenate([[0]*i+[0], elem]) for elem in self._get_all_actions(defendable[i+1:], n_defend)]+
                        [np.concatenate([[0]*i+[1], elem]) for elem in self._get_all_actions(defendable[i+1:], n_defend-1)])
            else:
                continue
        
        return []
    
    def get_all_actions(self, observation)->List[np.ndarray]:
        adj_mat, burned, defended = observation

        defendable = np.asarray(np.logical_and(
            np.any(adj_mat[burned], axis=0),
            np.logical_not(np.logical_or(defended, burned)),
        ), dtype=np.bool).reshape(-1)

        return self._get_all_actions(defendable, self.n_defend)
    
    def step(self, timestep) -> np.ndarray:
        
        if key not in self.policy:
            self.policy[(burned, defended)] = random.choice(self.get_all_actions(timestep.observation))
        
        return self.policy[(burned, defended)]