import gym
import matplotlib.pyplot as plt
import numpy as np
import mutliprocessing
from a2c import A2CAgent

seed = 112

env = gym.make('LunarLander-v2')
env.seed(seed)

# get the dimensions of the space
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = A2CAgent(theta = .0001, learning_rate= .002, discount=.99, actions= action_size, space= state_size)

episodes = 100
scores = {}

# play the game, each iteration is an episode
for i in range(episodes):
    done = False
    # score keeping
    total_rewards = 0

    # reset the environment
    observation = env.reset()
    observation = np.reshape(observation, [1, state_size])

    while not done:
        # get an action
        action = agent.next_action(observation)
        # get the rewards and info on that action
        observe, reward, done, info = env.step(action)
        # the agent learns given that data
        agent.learn(observation, action, reward, observe, done)
        # append the reward
        total_rewards += reward

        observation = observe
    scores[i] = total_rewards

print(scores)
