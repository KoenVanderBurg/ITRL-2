import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        action_values = np.zeros((n_states, n_actions))                    #initialize zero 2-d array [n_states * n_actions]
        self.action_values = action_values
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        pass

    def select_action(self, state: int) -> int:
        rand_number = np.random.rand()                            
        
        if rand_number > self.epsilon:                                      # 1-epsilon chance that greedy action gets returned
            return np.argmax(self.action_values[state])
        else:                                                               # pick other non-greedy action which has chance epsilon/3 , if greedy action is selected -> pick random action again
            action = np.random.choice(self.n_actions)
            while action == np.argmax(self.action_values[state]):
                action = np.random.choice(self.n_actions)
            return action

        
    def update(self, state, next_state, action, reward):
        self.action_values[state][action] = self.action_values[state][action] + self.alpha * ( reward + 1 * np.max(self.action_values[next_state]) - self.action_values[state][action]) # update rule
        return self.action_values

    def greedy_plot(self) -> np:
        Q = self.action_values                                             # returning the action_values for making the Greedy paths
        return(Q)
        


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        action_values = np.zeros((n_states, n_actions))       
        self.action_values = action_values
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        pass
        
    def select_action(self, state):
        rand_number = np.random.rand()
        action = np.argmax(self.action_values[state])

        if rand_number > self.epsilon:
            return action
        else:
            secondary_action = np.random.choice(self.n_actions)
            while secondary_action == action:
                secondary_action = np.random.choice(self.n_actions)
            return secondary_action
        
    def update(self, state, action, reward, next_state, next_action):     # update rule differs from Q-learning since it looks at the next state and next_action
        self.action_values[state][action] = self.action_values[state][action] + self.alpha * ( reward + 1 * (self.action_values[next_state][next_action]) - self.action_values[state][action])
        return self.action_values  

    def greedy_plot(self) -> np:
        Q = self.action_values
        return(Q)
              
        

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha):
       action_values = np.zeros((n_states, n_actions))        
       self.action_values = action_values
       self.n_actions = n_actions
       self.n_states = n_states
       self.epsilon = epsilon
       self.alpha = alpha
       pass
        
    def select_action(self, state):
        rand_number = np.random.rand()
        action = np.argmax(self.action_values[state])
        
        if rand_number > self.epsilon:
          return action
        else:
          secondary_action = np.random.choice(self.n_actions)
          while secondary_action == action:
            secondary_action = np.random.choice(self.n_actions)
          return secondary_action
        
    def update(self, state, action, reward, next_state):

        exp_Q = 0.0                                                                 # initializing expected Q value

        for next_action in range(self.n_actions):
            if next_action == np.argmax(self.action_values[next_state]):            # if the greedy next action is found add this value and multiply it with weight
                chance:float = (1 - self.epsilon)
                exp_Q += self.action_values[next_state][next_action] * chance
            else:                                                                   # if the next action is not the greedy action -> add this value and multiply it with weight
                chance:float = (self.epsilon / 3)
                exp_Q += self.action_values[next_state][next_action] * chance
        
        
        self.action_values[state][action] = self.action_values[state][action] + self.alpha * ( reward + 1 *exp_Q - self.action_values[state][action])   #perform actual update of state value
             
        return self.action_values  

    def greedy_plot(self) -> np:
        Q = self.action_values
        return(Q)
        


            