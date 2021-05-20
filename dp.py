import argparse
import copy
import math
import pickle
import random

from networkx import grid_graph, to_numpy_matrix
import numpy as np

from config import config
from env import FireFighter
from agents.policy_iter import policy_iter_agent
from agents.utils import numpy_dict

def get_all_states(env, agent):
    states = []

    def _get_all_states(env, agent, timestep):
        nonlocal states
        begin_state = timestep.observation[1:]
        
        visited = False
        for state in states:
            if (np.all(state[0]==begin_state[0]) and 
                np.all(state[1]==begin_state[1])):
                
                visited=True
                break
        
        if not visited:
            states.append(begin_state)
            if timestep.last():
                env.reset()
                return
            actions = agent.get_all_actions(timestep.observation)
            for action in actions:
                env.set_state(begin_state)
                for possible_timestep in env.all_possible_env_states(action):
                    _get_all_states(env, agent, possible_timestep)
            
    _get_all_states(env, agent, env.reset())
    return states

def policy_evaluation(env, agent, value_func: dict):
    while True:
        max_change = 0
        for state in value_func:
            old_value = value_func[state]
            env.set_state(state)
            timestep = env.step(agent.policy[state])
            reward, next_state = timestep.reward, timestep.observation[1:]
            value_func[state] = reward + value_func[next_state]
            max_change = max(max_change, abs(value_func[state] - old_value))

        if max_change < 0.1:
            break
    return value_func


def policy_improvement(env, agent, value_func: dict):
    policy_stable = True

    for state in value_func:
        old_action = agent.policy[state]
        max_val_func = float("-inf")

        for action in agent.get_all_actions(env._observation()):
            env.set_state(state)
            timestep = env.step(action)
            reward, next_state = timestep.reward, timestep.observation[1:]
            if reward + value_func[next_state] > max_val_func:
                agent.policy[state] = action
                max_val_func = reward + value_func[next_state]

        if old_action != agent.policy[state]:
            policy_stable = False

    return value_func, policy_stable


def policy_iteration(env, agent):
    states = get_all_states(env, agent)
    value_func = numpy_dict()
    
    for state in states:
        value_func[state] = np.random.randint(-3, 0)

    for state in states:
        env.set_state(state)
        agent.policy[state] = random.choice(agent.get_all_actions(env._observation()))

    while True:
        value_func = policy_evaluation(env, agent, value_func)
        print("Updated value function: ", value_func)
        value_func, policy_stable = policy_improvement(env, agent, value_func)
        if policy_stable:
            break

    return value_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write_file",
        type=str,
        required=True,
        help="File to write pickled value function and policy to",
    )
    args = parser.parse_args()

    burning_graph_env = FireFighter(
        config["adj_mat"], config["initial_fire"], config["burn_prob"]
    )
    ff_agent = policy_iter_agent(config["n_defend"])

    learned_value_func = policy_iteration(burning_graph_env, ff_agent)

    with open(args.write_file, "w+") as f:
        pickle.dump((learned_value_func, ff_agent.policy), f)
