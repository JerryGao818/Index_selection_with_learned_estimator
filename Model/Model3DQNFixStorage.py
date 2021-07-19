import numpy as np
import torch
import torch.nn as nn
from itertools import count
import torch.optim as optim
import os
import sys
import torch.nn.functional as F
import sys
sys.path.append('..')
import Enviornment.Env3DQNFixStorage as env
from tensorboardX import SummaryWriter

import random
import matplotlib.pyplot as plt
import math
import time
import json

from Model import PR_Buffer as BufferX

os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2'
device = torch.device('cuda:0')
script_name = os.path.basename(__file__)
directory = './exp_' + script_name + "_optimizer16" + '/'


class NN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            # nn.Sigmoid()
        )

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, state):
        actions = self.layers(state)
        return actions


class DQN:
    def __init__(self, workload, action, index_mode, conf):
        self.conf = conf
        self.workload = workload
        self.action = action
        self.index_mode = index_mode

        self.state_dim = len(action) + 2
        self.action_dim = len(action)

        self.actor = NN(self.state_dim, self.action_dim).to(device)
        self.actor_target = NN(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), conf['LR'])

        self.replay_buffer = BufferX.PrioritizedReplayMemory(self.conf['MEMORY_CAPACITY'], self.conf['LEARNING_START']) # ReplayBuffer(conf['MEMORY_CAPACITY'], conf['LEARNING_START'])  #   #

        # some monitor information
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.index_mode = index_mode
        self.actor_loss_trace = list()

        # environment
        self.envx = env.Env(self.workload, self.action, self.index_mode)

        # store the parameters
        self.writer = SummaryWriter(directory)

        self.learn_step_counter = 0

    def select_action(self, t, state):
        if not self.replay_buffer.can_update():
            print('init random selection: ', end='')
            action = np.random.randint(0, len(self.action))
            # action = [action]
            return action
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        state = state.to(device)
        if np.random.randn() <= self.conf['EPISILO']:  #*(t/MAX_STEP):  # greedy policy
            action_value = self.actor.forward(state)
            current_max = -100000
            action = -1
            for i in range(len(action_value[0])):
                if action_value[0][i] > current_max and self.envx.currenct_index[i] < 0.9:
                    action = i
                    current_max = action_value[0][i]
            if action == -1 or current_max <= 0:
                print('negative random select: ', end='')
                action = np.random.randint(0, len(self.action))
            return action
        else:  # random policy
            print('true random select: ', end='')
            action = np.random.randint(0, len(self.action))
            return action

    def select_best_action(self, t, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        state = state.to(device)
        action_value = self.actor.forward(state)
        current_max = -100000
        action = -1
        for i in range(len(action_value[0])):
            if action_value[0][i] > current_max and self.envx.currenct_index[i] < 0.9:
                action = i
                current_max = action_value[0][i]
        if action == -1:
            action = np.random.randint(0, len(self.action))
        return action

    def _sample(self):
        batch, idx = self.replay_buffer.sample(self.conf['BATCH_SIZE'])
        x, y, u, r, d = [], [], [], [], []
        for _b in batch:
            x.append(np.array(_b[0], copy=False))
            y.append(np.array(_b[1], copy=False))
            u.append(np.array(_b[2], copy=False))
            r.append(np.array(_b[3], copy=False))
            d.append(np.array(_b[4], copy=False))
        return idx, np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


    def update(self):
        if self.learn_step_counter % self.conf['Q_ITERATION'] == 0:
            print('Q-target updated.')
            self.actor_target.load_state_dict(self.actor.state_dict())
        self.learn_step_counter += 1
        for it in range(self.conf['U_ITERATION']):
            idxs, x, y, u, r, d = self._sample()
            state = torch.FloatTensor(x).to(device)
            action = torch.LongTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)
            action = action.unsqueeze(1)
            q_eval = self.actor(state).gather(1, action)
            q_next = self.actor_target(next_state).detach()
            q_target = reward + (1-done)*self.conf['GAMMA'] * q_next.max(1)[0].view(self.conf['BATCH_SIZE'], 1)
            actor_loss = F.mse_loss(q_eval, q_target)
            error = torch.abs(q_eval - q_target).data.cpu().numpy()
            for i in range(self.conf['BATCH_SIZE']):
                idx = idxs[i]
                self.replay_buffer.update(idx, error[i][0])

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_loss_trace.append(actor_loss)

    def save(self):
        print('====== Model Saved ======')
        torch.save(self.actor_target.state_dict(), directory + 'dqn.pth')
        self.replay_buffer.save(directory + 'PR_buffer.pickle')


    def load(self):
        print('====== Model Loaded ======')
        self.actor.load_state_dict(torch.load(directory + 'dqn.pth'))
        self.replay_buffer.load_memory(directory + 'PR_buffer.pickle')

    def train(self, load, __x):
        if load:
            self.load()
        is_first = True
        self.envx.max_size = __x*134815744
        current_best_time = 10000000
        # current_best_reward = 0
        current_best_index = None
        current_best_index_s = 0
        for ep in range(self.conf['EPISODES']):
            print("======"+str(ep)+"=====")
            state = self.envx.reset()
            print('init sum: ', self.envx.init_cost_sum)
            rewards = []
            t_r = 0
            last_input_state = None
            last_next_state = None
            last_action = None
            last_reward = None
            for t in count():
                action = self.select_action(t, state)
                next_state, reward, done = self.envx.step(action)
                t_r += reward
                rewards.append(t_r)
                if not done:
                    last_input_state = state
                    last_next_state = next_state
                    last_action = action
                    last_reward = reward
                if done:
                    if last_reward is None:
                        break
                    print('current solution: ', end='')
                    for _i, _idx in enumerate(self.envx.index_trace_overall[-1]):
                        if _idx == 1.0:
                            print(self.envx.candidates[_i], end=', ')
                    print(' ')
                    if  self.envx.cost_trace_overall[-1] > current_best_time:
                        # current_best_reward = t_r
                        current_best_time = self.envx.cost_trace_overall[-1]
                        current_best_index = self.envx.index_trace_overall[-1]
                        current_best_index_s = self.envx.storage_trace_overall[-1]
                        print('new best saved.')
                    self.replay_buffer.add(1.0, (last_input_state.tolist(), last_next_state.tolist(), last_action, last_reward, np.float(done)))
                    if self.replay_buffer.can_update():
                        if is_first:
                            print("first:", ep)
                            is_first = False
                        self.update()
                    break
                state = next_state
            self.save()
        print("===========")
        plt.figure(__x)
        x = range(len(self.envx.cost_trace_overall))
        y2 = [math.log(a, 10) for a in self.envx.cost_trace_overall]
        json.dump(self.envx.cost_trace_overall, open(self.conf['NAME']+'result.json', 'w'))
        plt.plot(x, y2, marker='x')
        plt.savefig(self.conf['NAME'] + "freq.png", dpi=120)
        plt.clf()
        plt.close()
        print('best time: ', current_best_time)
        return current_best_index, current_best_index_s


