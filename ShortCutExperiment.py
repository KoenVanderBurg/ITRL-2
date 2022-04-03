import re
from statistics import mean
from typing_extensions import final
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth


def run_repetitions_QA( n_actions, n_episodes, alpha , epsilon, n_rep, n_states, all_rewards,rewards, pi,rep):

    for episode in range(n_episodes):                                        #for each repetition and episode
        env = ShortcutEnvironment()                                             # initialize environment

        while env.done() == False:
            c_state = env.state()                                                #current state
            a = pi.select_action(c_state)                                        #select action
            r = env.step(a)                                                      #make move -> get reward
            Q = pi.update(c_state, env.state(),a,r)                                  #update policy
            rewards[episode] += r                                                #store reward into rewards

        all_rewards[rep][episode] = rewards[episode]
    
    return (np.average(all_rewards, 0), Q)                                             #return the average over all the rewards

def experiment( n_actions, n_episodes, n_rep, epsilon, smoothing_window, n_states):

    # Assignment 1: e-greedy
    plot = LearningCurvePlot('Learning Curves Q-Learning Agent')
    plot2 = LearningCurvePlot('Learning Curves Q-Learning Agent Final Run')
    alphas = [0.01, 0.1, 0.5, 0.9]
    best_return_Q = 0.0
    best_run_Q = [0 for x in range(n_rep)]                                   
    average_returns_Q = []
    all_rewards = np.zeros((n_rep, n_episodes))                    #2d-array to store all the rewards in of the n_repetitions * n_episodes.
    for alpha in alphas:
        for rep in range(n_rep):                                                     #for each repetition:
            print(f"Now loading alpha: {alpha}, done {rep} / {n_rep} reps", end = '\r')
            rewards = np.zeros(n_episodes)                                             # initialize rewards
            pi = QLearningAgent(n_actions = n_actions, n_states = n_states,
                                alpha = alpha, epsilon = epsilon)                            # initialize policy  
            if rep == (n_rep -1):
                final_run_tuple = run_repetitions_QA( n_actions, n_episodes, alpha, epsilon, n_rep, n_states, all_rewards,rewards,pi,rep)
                final_run = final_run_tuple[0]
                plot2.add_curve((final_run), label= f'{alpha = }')

            rewards_tuple = run_repetitions_QA( n_actions, n_episodes, alpha, epsilon, n_rep, n_states, all_rewards,rewards,pi,rep)
            rewards = rewards_tuple[0]
            average_returns_Q.append(np.average(rewards)) 


        plot.add_curve((average_returns_Q))
        plot.add_curve(smooth(rewards,smoothing_window), label = f'{alpha = }')

    plot.save("QA_2_average.png")
    plot2.save("QA_2_final_run.png")
    print_greedy_actions(final_run_tuple[1])

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

    experiment( n_actions = n_actions, n_episodes = n_episodes, n_rep = n_rep, epsilon = epsilon, n_states = n_states, smoothing_window = smoothing_window)

