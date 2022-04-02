import re
from statistics import mean
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth

def run_repetitions_QA (n_actions, n_episodes, alpha , epsilon, n_rep):
    all_rewards = np.zeros((n_episodes,n_actions))                    #2d-array to store all the rewards in of the n_repetitions * n_episodes.

    for rep in range(n_rep):                                                     #for each repetition:
        env = ShortcutEnvironment()                                             # initialize environment  
        n_states = env.state_size()  
        pi = QLearningAgent(n_actions = n_actions, n_states = n_states,
                             alpha = alpha, epsilon = epsilon)                            # initialize policy
        rewards =np.zeros(n_episodes)                                            # initialize rewards
        for episode in range(n_episodes):                                        #for each repetition and timestep
            start = env.reset()
            while env.isdone == False:
                a = pi.select_action(env.state())                                    #select action
                r = env.step(a)                                                      #fetch reward
                pi.update(env.state(),a,r)                                                       #update policy
                rewards[episode] = r                                                #store reward into rewards

        all_rewards[episode] = rewards
    return np.average(all_rewards, 0)                                      #return the average over all the rewards

def experiment(n_actions, n_episodes, n_rep, alpha, epsilon, smoothing_window):

    # Assignment 1: e-greedy
    plot = LearningCurvePlot('Learning Curves Epsilon Greedy')
    epsilons = [ 0.01]
    best_return_greedy = 0.0
    best_run_greedy = [0 for x in range(n_rep)]                                   
    average_returns_greedy = []
    for epsilon in epsilons:
        rewards = run_repetitions_QA(n_actions, n_episodes, alpha, epsilon, n_rep)
        average_reward = np.average(rewards)                                            #getting the average rewards over all the rewards.
        average_returns_greedy.append(average_reward)
        if average_reward > best_return_greedy:                                         #if the average_reward is better than the so far best, the average_reward becomes the new best.
            best_return_greedy = average_reward
            best_run_greedy = rewards                                                   #the accompanying best run gets stored as well.


        plot.add_curve(rewards)
        plot.add_curve(smooth(rewards,smoothing_window), label = f'{epsilon = }')
  
    plot.save("QA_1.png")

if __name__ == '__main__':
    # experiment settings
    n_episodes = 1000
    n_rep = 100
    alpha = 0.1
    epsilon = 0.1
    smoothing_window = 31
    n_actions = 4

    experiment(n_actions = n_actions, n_episodes = n_episodes,n_rep = n_rep, alpha=alpha, epsilon = epsilon, smoothing_window=smoothing_window)

