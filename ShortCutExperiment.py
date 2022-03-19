import re
from statistics import mean
import numpy as np
from ShortCutEnvironment import ShortcutEnvironment
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth

def run_repetitions_QA(n_actions, n_timesteps, n_repetitions,epsilon):
    

    
    all_rewards = np.zeros((n_repetitions,n_timesteps))                    #2d-array to store all the rewards in of the n_repititions * n_timesteps.

    for repetition in range(n_repetitions):                                #for each repetition:

        env = BanditEnvironment(n_actions=n_actions)                            # initialize environment    
        pi = EgreedyPolicy(n_actions=n_actions)                                 # initialize policy
        rewards =np.zeros(n_timesteps)                                          # initialize rewards
        for timestep in range(n_timesteps):                                #for each repetition and timestep:
            a = pi.select_action(epsilon)                                        #select action
            r = env.act(a)                                                       #sample reward
            pi.update(a,r)                                                       #update policy
            rewards[timestep] = r                                                #store reward into rewards

        all_rewards[repetition] = rewards

    return np.average(all_rewards, 0)                                      #return the average over all the rewards

def experiment():

    # Assignment 1: e-greedy
    plot = LearningCurvePlot('Learning Curves Epsilon Greedy')
    epsilons = [ 0.01,0.05,0.1,0.25]
    best_return_greedy = 0.0
    best_run_greedy = [0 for x in range(n_timesteps)]                                   
    average_returns_greedy = []
    for epsilon in epsilons:
        rewards = run_repetitions_eg(n_actions, n_timesteps, n_repetitions, epsilon)
        average_reward = np.average(rewards)                                            #getting the average rewards over all the rewards.
        average_returns_greedy.append(average_reward)
        if average_reward > best_return_greedy:                                         #if the average_reward is better than the so far best, the average_reward becomes the new best.
            best_return_greedy = average_reward
            best_run_greedy = rewards                                                   #the accompanying best run gets stored as well.


        plot.add_curve(rewards)
        plot.add_curve(smooth(rewards,smoothing_window), label = f'{epsilon = }')
  
    plot.save("EG_4.png")

if __name__ == '__main__':
    # experiment settings
    n_episodes = 1000
    n_rep = 100
    alpha = 0.1
    epsilon = 0.1
    smoothing_window = 31

    experiment(n_episodes = n_episodes,n_rep = n_rep, alpha=alpha, epsilon = epsilon, smoothing_window=smoothing_window)

