import re
from statistics import mean
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth

def run_repetitions_QA (n_actions, n_episodes, alpha , epsilon, n_rep, n_states):
    all_rewards = np.zeros((n_rep, n_episodes))                    #2d-array to store all the rewards in of the n_repetitions * n_episodes.

    for rep in range(n_rep):                                                     #for each repetition:
        rewards = np.zeros(n_episodes)                                             # initialize rewards
        pi = QLearningAgent(n_actions = n_actions, n_states = n_states,
                             alpha = alpha, epsilon = epsilon)                            # initialize policy  

        for episode in range(n_episodes):                                        #for each repetition and episode
            env = ShortcutEnvironment()                                             # initialize environment

            while env.done() == False:
                c_state = env.state()                                                #current state
                a = pi.select_action(c_state)                                        #select action
                r = env.step(a)                                                      #make move -> get reward
                pi.update(c_state, env.state(),a,r)                                  #update policy
                rewards[episode] += r                                                #store reward into rewards

            all_rewards[rep][episode] = rewards[episode]
    Q = QLearningAgent.greedy_plot()  
    print_greedy_actions(Q) 
    return np.average(all_rewards, 0)                                                   #return the average over all the rewards

def experiment(n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states):

    # Assignment 1: e-greedy
    plot = LearningCurvePlot('Learning Curves Q-Learning Agent')
    alphas = [0.01, 0.1, 0.5, 0.9]
    best_return_greedy = 0.0
    best_run_greedy = [0 for x in range(n_rep)]                                   
    average_returns_greedy = []
    for alpha in alphas:
        rewards = run_repetitions_QA(n_actions, n_episodes, alpha, epsilon, n_rep, n_states)
        average_reward = np.average(rewards)                                            #getting the average rewards over all the rewards.
        average_returns_greedy.append(average_reward)
        if average_reward > best_return_greedy:                                         #if the average_reward is better than the so far best, the average_reward becomes the new best.
            best_return_greedy = average_reward
            best_run_greedy = rewards                                                   #the accompanying best run gets stored as well.


        plot.add_curve(rewards)
        plot.add_curve(smooth(rewards,smoothing_window), label = f'{alpha = }')
  
    plot.save("QA_1.png")

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


if __name__ == '__main__':
    # experiment settings
    n_episodes = 1000
    n_rep = 100
    alpha = 0.1
    epsilon = 0.1
    smoothing_window = 31
    n_actions = 4
    n_states = 144

    experiment(n_actions = n_actions, n_episodes = n_episodes, n_rep = n_rep, epsilon = epsilon, n_states = n_states, smoothing_window = smoothing_window)

