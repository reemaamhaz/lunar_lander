import os
import numpy as np
from pandas import get_dummies
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

'''
    This class introduces a Actor-Critic Algorithm creates an Actor that takes actions
    and has the Critic evaluate it correct the actions taken over time.
'''
class A2CAgent(object):
    def __init__(self, theta, learning_rate, discount, actions, space):
        '''
            This function initializes our agent with its default values and 
            creates an actor, critic, and policy.
            Parameters: theta, learning_rate, discount, actions
            Returns: None
        '''
        # set the learning rate, discount
        self.theta = theta
        self.learning_rate = learning_rate
        self.discount = discount
        
        self.actions = actions
        self.space = space

        #creat an actor and critic
        network = NeuralNetwork(theta,learning_rate, actions, space)
        self.actor = network.build_actor()
        self.critic = network.build_critic()
        
    def next_action(self, observe):
        # get the next action
        state = observe[np.newaxis, :]
        probabilities = self.actor.predict(state, batch_size = 1).flatten()
        action = np.random.choice(self.action_list,1, p = probabilities)[0]

        return action

    def learn(self, info, action, reward, state, done):
        # learn based on what maximizes the reward
        target = state[np.newaxis, :]
        advantages = info[np.newaxis, :]
        
        val_1 = self.critic.predict(state)[0]
        val_2 = self.critic.predict(info)[0]

        if done:
            advantages[0][action] = reward - val_1
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount * val_2 - val_1
            target[0][0] = reward + self.discount * val_2

        self.actor.fit(state, advantages, verbose = 0)
        self.critic.fit(state, target, verbose = 0)

        

class NeuralNetwork(object):
    def __init__(self,theta, lr, actions, space):
        '''
            This function sets up the actor crtic network using two layers.
            Parameters: None
            Returns: None
        '''
        self.lr = lr
        self.theta = theta
        self.value_size = actions
        self.state_size = space

    def mse(self, y, y_hat):
        # get the absolute value of the difference in means squared
        return np.abs(np.mean(y) - np.mean(y_hat))**2 

    def squared_loss(self, y_obs, y_hat):
        # calculate the squared loss of the true y vs the predicted y
        return (y_obs - y_hat)**2  

    def build_actor(self):
        '''
            Creates the actor network with softmax.
        '''
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        actor.add(Dense(self.value_size, activation='softmax',
                         kernel_initializer='he_uniform'))
        actor.compile(loss =squared_loss, optimizer=Adam(lr = self.theta))
        return actor

    def build_critic(self):
        '''
            Creates the critic network
        '''
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        
        critic.compile(loss = mse, optimizer=Adam(lr = self.lr))
        return critic