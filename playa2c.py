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
    
# import gym, os
# from a2c import Agent
# from gym import wrappers
# import numpy as np

# if __name__ == '__main__':
#     agent = Agent(alpha=0.00001, beta=0.00005)

#     env = gym.make('LunarLander-v2')
#     score_history = []
#     num_episodes = 2000

#     for i in range(num_episodes):
#         done = False
#         score = 0
#         observation = env.reset()
#         while not done:
#             action = agent.choose_action(observation)
#             observation_, reward, done, info = env.step(action)
#             agent.learn(observation, action, reward, observation_, done)
#             observation = observation_
#             score += reward

#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])
#         print('episode: ', i,'score: %.2f' % score,
#               'avg score %.2f' % avg_score)
