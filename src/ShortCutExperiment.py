from os import environ
import re
from statistics import mean
from typing_extensions import final
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth


def run_repetitions_QA( n_episodes, all_rewards,rewards, pi,rep, environment):

    for episode in range(n_episodes):                                           
        env = environment()                                                      # initialize environment

        while env.done() == False:
            c_state = env.state()                                                # current state
            a = pi.select_action(c_state)                                        # select action
            r = env.step(a)                                                      # make move -> get reward
            q = pi.update(c_state, env.state(),a,r)                              # update policy -> store action_values
            rewards[episode] += r                                                # store reward into rewards

        all_rewards[rep][episode] = rewards[episode]                             # update all rewards
    
    return (np.average(all_rewards, 0), q)                                       # return the average over all the rewards

def run_repetitions_SARSA( n_episodes, all_rewards,rewards, pi,rep,environment):

    for episode in range(n_episodes):                                        
        env = environment()                                             
        a = pi.select_action(env.state())

        while env.done() == False:
            c_state = env.state()                                                # current state
            r = env.step(a)                                                      # make move -> get reward
            rewards[episode] += r                                                # store reward into rewards 
            na = pi.select_action(env.state())                                   # storing the next_action
            q = pi.update(c_state, a, r, env.state(), na)                        # update policy, state, action, reward, next_state, next_action
            a = na                                                               # action is next_action

        all_rewards[rep][episode] = rewards[episode]
    
    return (np.average(all_rewards, 0), q)                                       # return the average over all the rewards

def run_repetitions_EXP_SARSA( n_episodes, all_rewards,rewards, pi,rep,environment):

    for episode in range(n_episodes):                                       
        env = environment()                                                     
        a = pi.select_action(env.state())                                        # pre-select an action before entering loop

        while env.done() == False:
            c_state = env.state()                                                
            r = env.step(a)                                                      
            rewards[episode] += r                                                 
            na = pi.select_action(env.state())                                   
            q = pi.update(c_state, a, r, env.state())                        
            a = na                                                               

        all_rewards[rep][episode] = rewards[episode]

    return (np.average(all_rewards, 0), q)                                             

def experiment_EXP_SARSA(n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states, environment, alpha = None):

    plot = LearningCurvePlot('Learning Curves Expected SARSA Agent')

    if alpha == None:
        alphas = [0.01, 0.1, 0.5, 0.9]
    else:
        alphas = [alpha]
         
    average_returns_Q = []
    all_rewards = np.zeros((n_rep, n_episodes))                                                                    # 2d-array to store all the rewards in of the n_repetitions * n_episodes.

    for alpha in alphas:                                                                                           # for each alpha
        for rep in range(n_rep):                                                                                   # for each repetition:
            print(f"EXP-SARSA Now loading alpha: {alpha}, done {rep + 1} / {n_rep} reps", end = '\r')              # progress bar
            rewards = np.zeros(n_episodes)                                                                         # initialize rewards
            pi = ExpectedSARSAAgent(n_actions = n_actions, n_states = n_states,
                                alpha = alpha, epsilon = epsilon)                                                  # initialize policy  

            if rep == (n_rep -1):
                print('\n')

            rewards_tuple = run_repetitions_EXP_SARSA( n_episodes, all_rewards,rewards,pi,rep, environment)        # do repetitions 
            average_returns_Q.append(np.average(rewards_tuple[0]))                                                 # append the average reward


        plot.add_curve((average_returns_Q))
        plot.add_curve(smooth(rewards,smoothing_window), label = f'{alpha = }')

    if len(alphas) == 1:
        if environment == WindyShortcutEnvironment:
            print(f"EXP-SARSA ~ Windy (alpha: {alpha}) plot for {n_episodes} episodes and {n_rep} rep")
        else:
            print(f"EXP-SARSA (alpha: {alpha}) plot for {n_episodes} episodes and {n_rep} rep")
        print_greedy_actions(rewards_tuple[1])
    else:
       plot.save("EXP-SARSA_1_average.png")
 
def experiment_SARSA(n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states, environment, alpha = None):

    plot = LearningCurvePlot('Learning Curves SARSA Agent')

    if alpha == None:
        alphas = [0.01, 0.1, 0.5, 0.9]
    else:
        alphas = [alpha]
         
    average_returns_Q = []
    all_rewards = np.zeros((n_rep, n_episodes))                    

    for alpha in alphas:
        for rep in range(n_rep):                                                   
            print(f"SARSA Now loading alpha: {alpha}, done {rep + 1} / {n_rep} reps", end = '\r')
            rewards = np.zeros(n_episodes)                                          
            pi = SARSAAgent(n_actions = n_actions, n_states = n_states,
                                alpha = alpha, epsilon = epsilon)                            
            if rep == (n_rep -1):
                print('\n')

            rewards_tuple = run_repetitions_SARSA( n_episodes, all_rewards,rewards,pi,rep, environment)
            average_returns_Q.append(np.average(rewards_tuple[0])) 


        plot.add_curve((average_returns_Q))
        plot.add_curve(smooth(rewards,smoothing_window), label = f'{alpha = }')

    if len(alphas) == 1:
        if environment == WindyShortcutEnvironment:
            print(f"SARSA ~ Windy (alpha: {alpha}) plot for {n_episodes} episodes and {n_rep} rep")
        else:
            print(f"SARSA (alpha: {alpha}) plot for {n_episodes} episodes and {n_rep} rep")
        print_greedy_actions(rewards_tuple[1])
    else:
        plot.save("SARSA_1_average.png")

def experiment_QA( n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states, environment, alpha = None, ):
    
    plot = LearningCurvePlot('Learning Curves Q-Learning Agent')

    if alpha == None:
        alphas = [0.01, 0.1, 0.5, 0.9]
    else:
        alphas = [alpha]

    average_returns_Q = []
    all_rewards = np.zeros((n_rep, n_episodes))                  

    for alpha in alphas:
        for rep in range(n_rep):                                                    
            print(f"QA: Now loading alpha: {alpha}, done {rep + 1} / {n_rep} reps", end = '\r')
            rewards = np.zeros(n_episodes)                                            
            pi = QLearningAgent(n_actions = n_actions, n_states = n_states,
                                alpha = alpha, epsilon = epsilon)                              
            if rep == (n_rep -1):
                print("\n")

            rewards_tuple = run_repetitions_QA(n_episodes, all_rewards,rewards, pi,rep,environment)
            average_returns_Q.append(np.average(rewards_tuple[0])) 


        plot.add_curve((average_returns_Q))
        plot.add_curve(smooth(rewards,smoothing_window), label = f'{alpha = }')

    if len(alphas) == 1:
        if environment == WindyShortcutEnvironment:
            print(f"QA ~ Windy (alpha: {alpha}) plot for {n_episodes} episodes and {n_rep} rep")
        else:
            print(f"QA (alpha: {alpha}) plot for {n_episodes} episodes and {n_rep} rep")
        print_greedy_actions(rewards_tuple[1])
    else:
        plot.save("QA_1_average.png")
    

def print_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12,12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions==0] = '^'
    print_string[greedy_actions==1] = 'v'
    print_string[greedy_actions==2] = '<'
    print_string[greedy_actions==3] = '>'
    print_string[np.max(Q, 1).reshape((12, 12))==0] = ' '
    line_breaks = np.zeros((12,1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    print(print_string.tobytes().decode('utf-8'))

def run_experiment( n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states):
    #experiment_QA(n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states, environment =  ShortcutEnvironment)
    #experiment_QA(n_actions, 10000, 1, epsilon, smoothing_window, n_states, environment = ShortcutEnvironment, alpha = 0.1)
    experiment_QA(n_actions, 10000, 1, epsilon, smoothing_window, n_states, environment = WindyShortcutEnvironment, alpha = 0.1,)
    experiment_SARSA(n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states, environment = ShortcutEnvironment)
    experiment_SARSA(n_actions, 10000, 1, epsilon, smoothing_window, n_states, environment = ShortcutEnvironment, alpha = 0.1)
    experiment_SARSA(n_actions, 10000, 1, epsilon, smoothing_window, n_states, environment = WindyShortcutEnvironment, alpha = 0.1)
    #experiment_EXP_SARSA(n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states, environment = ShortcutEnvironment)
    #experiment_EXP_SARSA(n_actions, 10000, 1, epsilon, smoothing_window, n_states, environment = ShortcutEnvironment, alpha = 0.1)

    
if __name__ == '__main__':
    # experiment settings
    n_episodes = 1000
    n_rep = 100
    epsilon = 0.1
    smoothing_window = 31
    n_actions = 4
    n_states = 144

    run_experiment( n_actions = n_actions, n_episodes = n_episodes, n_rep = n_rep, epsilon = epsilon, n_states = n_states, smoothing_window = smoothing_window)

