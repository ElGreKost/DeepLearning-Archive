import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from vizdoom.gym_wrapper.gym_env_defns import VizdoomScenarioEnv

import experience_replay
from image_preprocessing import PreprocessImage


# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):  # The input_image is (1x80x80)
    def __init__(self, number_actions, ):
        super(CNN, self).__init__()
        self.convolutional1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        torch.nn.init.xavier_normal_(self.convolutional1[0].weight)
        # self.convolutional1.apply(init_cnn)
        self.convolutional2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        torch.nn.init.xavier_normal_(self.convolutional2[0].weight)

        self.convolutional3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        torch.nn.init.xavier_normal_(self.convolutional3[0].weight)

        self.fc1 = nn.Sequential(
            # nn.Linear(self.count_neurons((1, 80, 80)), out_features=40),
            nn.LazyLinear(40),
            nn.Dropout(),
            nn.ReLU()
        )
        torch.nn.init.xavier_normal_(self.convolutional3[0].weight)
        # self.fc1 = nn.Sequential(nn.Linear(out_features=40), nn.Dropout(), nn.ReLU())
        self.fc2 = nn.Linear(40, number_actions)

        # self.apply_init(init_cnn) # make weights xavier_uniform

    def count_neurons(self, image_dim):  # If we don't use LazyLinear
        x = Variable(torch.rand(1, *image_dim))
        # kernel = 3, stride = 2
        x = self.convolutional1(x)
        x = self.convolutional2(x)
        x = self.convolutional3(x)
        # return x.data.view(1,-1).size(1)
        return x.data.view(1, -1, antialias=True).size(1)

    def forward(self, x):
        x = self.convolutional1(x)
        x = self.convolutional2(x)
        x = self.convolutional3(x)
        x = x.view(x.size(0), -1, antialias=True)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    # def apply_init(self, inputs, init=None):
    #     self.forward(*inputs)
    #     if init is not None:
    #         self.net.apply(init)


# Making the body
class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        # to get the action, we use softmax
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions


# Making the AI
class AI:
    def __init__(self, brain: CNN, body: SoftmaxBody):
        self.brain = brain
        self.body = body

    # The combination of the forward functions
    def __call__(self, inputs):  # with __call__ you can treat an AI object as a function
        inp = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.brain(inp)
        actions = self.body(output)
        return actions.data.numpy()  # tensor to array


doom_env = VizdoomScenarioEnv(
    scenario_file='deadly_corridor.cfg',  # Specify the path to your scenario file
    frame_skip=4,  # Adjust the frame skip if needed
    max_buttons_pressed=1,  # Adjust the max buttons pressed if needed
    render_mode='human'  # Set the render mode (either 'human' or 'rgb_array')
)

# Initialized the env here so that CNN can get the action_space length

doom_env = PreprocessImage(doom_env, width=80, height=80, grayscale=True)
# doom_env = gym.wrappers.ResizeObservation(doom_env, (80, 80))
# doom_env = gym.wrappers.GrayScaleObservation(doom_env)
doom_env = gym.wrappers.RecordVideo(doom_env, "videos/doom.mp4")
number_actions = doom_env.action_space.n

# Building the AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T=1.0)

ai = AI(brain=cnn, body=softmax_body)

# Setting up Xp Replay: We have accumulative target and reward on 10 steps instead of 1 like before
# Because we have eligibility trace
n_steps = experience_replay.NStepProgress(doom_env, ai, 10)
memory = experience_replay.ReplayMemory(n_steps=n_steps)


# Implementing Eligibility Trace
def eligibility_trace(batch):
    """
    The batch contains objects of type Step with ['state', 'action', 'reward', 'done'] format
    :param batch: contains input and targets4
    :return:
    """
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        inp = Variable(
            torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))  # state and done
        output = cnn(inp)  # Which is the prediction
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inp, dtype=np.float32)), torch.stack(targets)


# Making the moving average on 100 steps
class MA:
    # to keep track of the rewards mean during the training
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)


ma = MA(100)

# Train the AI
loss = nn.MSELoss()  # The general regression loss
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    # Each epoch will do 200 runs of 10 steps(from replay memory)
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_rewards = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch)), str(avg_rewards))

import os
import cv2
import glob
import gym
import vizdoom as vzd
import matplotlib.pyplot as plt
from collections import Counter
# from gym.wrappers import Monitor # Deprecated

doom_env = VizdoomScenarioEnv(
    scenario_file='deadly_corridor.cfg',  # Specify the path to your scenario file
    frame_skip=4,  # Adjust the frame skip if needed
    max_buttons_pressed=1,  # Adjust the max buttons pressed if needed
    render_mode='human'  # Set the render mode (either 'human' or 'rgb_array')
)

doom_env = PreprocessImage(doom_env, width=80, height=80, grayscale=True)
# doom_env = gym.wrappers.ResizeObservation(doom_env, (80, 80))
# doom_env = gym.wrappers.GrayScaleObservation(doom_env)
doom_env = gym.wrappers.RecordVideo(doom_env, "videos/doom.mp4")
action_num = doom_env.action_space.n
print("Number of possible actions: ", action_num)
state = doom_env.reset()
state, reward, done, info = doom_env.step(doom_env.action_space.sample())


