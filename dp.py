import argparse
import copy
import math
import pickle
import random

from networkx import grid_graph, to_numpy_matrix

from config import config
from env import FireFighter
from agents.policy_iter import policy_iter_agent

def get_all_states(env, agent):
    states = []
    
    def get_all_states(env, agent, timestep):
        nonlocal states
        begin_state = timestep.observation[1:]
        if begin_state not in states:
            states.append(begin_state)
            actions = agent.get_all_actions(timestep.observation)
            for action in actions:
                env.set_state(begin_state)
                timestep = env.step(action)
                if not timestep.last():
                    state = timestep.observation[1:]
                    states += get_all_states(env, agent, timestep)
        return states
    
    return get_all_states(env, agent, env.reset())

def policy_evaluation(env, agent, value_func: dict):        
    while True:
        max_change = 0
        for state in value_func:
            old_value = value_func[state]
            env.set_state(state)
            timestep = env.step(agent.policy[state])
            reward, next_state = timestep.reward, timestep.observation[1:]
            value_func[state] = reward+value_func[next_state]
            max_change = max(max_change, abs(value_func[state]-old_value))
        
        if max_change<0.1:
            break
    return value_func

def policy_improvement(env, agent, value_func: dict):
    policy_stable = True
    
    for state in value_func:
        old_action = agent.policy[state]
        max_val_func = float('-inf')
            
        for action in agent.get_all_actions(env._observation()):
            env.set_state(state)
            timestep = env.step(action)
            reward, next_state = timestep.reward, timestep.observation[1:]    
            if reward+value_func[next_state]>max_val_func:
                agent.policy[state] = action
                max_val_func = reward+value_func[next_state]
            
        if old_action!=agent.policy[state]:
            policy_stable=False
    
    return value_func, policy_stable

def policy_iteration(env, agent):
    states = get_all_states(env, agent)
    value_func = {state: np.random.randint(-3,0) for state in states}
    
    for state in states:
        env.set_state(state)
        agent.policy[state] = random.choice(agent.get_all_actions(env._observation()))
    
    while True:
        value_func, policy_stable = policy_improvement(env, agent, value_func)
        if policy_stable:
            break
    
    return value_func



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_file', type=str, required=True, help='File to write pickled value function and policy to')
    args = parser.parse_args()

    env = FireFighter(config['adj_mat'], config['initial_fire'], config['burn_prob'])
    agent = policy_iter_agent(config['n_defend'])
    
    value_func = policy_iteration(env, agent)
    
    with open(args.write_file, 'w+') as f:
        pickle.dump((value_func, agent.policy), f)
