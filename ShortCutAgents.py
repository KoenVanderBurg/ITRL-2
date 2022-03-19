import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, next_state):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.next_state = next_state
        action_values = [[0 for i in range (self.n_states)] for j in range (self.n_actions)]        #initialize zero 2-d array [n_states * n_actions]
        pass
        
    def select_action(self, state):

        prob = [[0 for x in range (self.n_states)] for j in range(self.n_actions)]                   #create probabilities list.                   
        
        for action in range(self.n_actions):
            if action == np.argmax(self.action_values[state]):
                prob[state][action] = (1 - self.epsilon)                                               #assign highest mean of action the highest probability. 
            else:
                prob[state][action] = (self.epsilon/(self.n_actions - 1))                              #assign rest of probability values to rest of the actions.
        
        return(np.random.choice(self.n_actions, p = prob))                                          #return a random action based on the probabilities.
        
    def update(self, state, action, reward):
        self.action_values[state][action] = self.action_values[state][action] + self.alpha * ( reward + 1 * np.argmax(self.action_values[self.next_state]) - self.action_values[state][action])
        pass

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass
        
    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions)) # Replace this with correct action selection
        return a
        
    def update(self, state, action, reward):
        # TO DO: Add own code
        pass