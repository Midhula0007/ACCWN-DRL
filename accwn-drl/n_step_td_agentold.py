import argparse
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno
from actor_critic_continuous import NStepAgent
from plotting_utility import plotLearning

import math
import csv
import matplotlib.pyplot as plt
import collections

import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SharedAdam(T.optim.Adam):
    """docstring for SharedAdam"""
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.99), eps = 1e-8, weight_decay = 0):
        super(SharedAdam, self).__init__(params, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        print("INIT in SharedAdam")
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)
                
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    """docstring for ActorCritic"""
    def __init__(self, input_dims, n_actions, gamma = 0.99):
        super(ActorCritic, self).__init__()
        print("INIT in ActorCritic")
        self.input_dims = input_dims
        self.gamma = gamma
        self.n_actions = n_actions
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128,1)

        self.rewards = []
        self.actions = []
        self.states = []
    
    def remember(self, state, action, reward):
        print("remember in ActorCritic")
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def clear_memory(self):
        print("clear_memory in ActorCritic")
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        print("forward in ActorCritic")
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))

        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    def calc_R(self, done):
        print("calc_R in ActorCritic")
        states = T.tensor(self.states, dtype = T.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        print("calc_loss in ActorCritic")
        states = T.tensor(self.states, dtype = T.float)
        actions = T.tensor(self.actions, dtype = T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim = 1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss+actor_loss).mean()

        return total_loss

    def choose_action(self, observation):
        #print("choose action in ActorCritic")
        #state = T.tensor([observation], dtype = T.float)
        pi, v = self.forward(observation)
        #print("Pi - : {}".format(pi))
        #print("v - : {}".format(v))
        probs = T.softmax(pi, dim = 1)
        #print("probs - ", probs)
        dist = Categorical(probs)
        #print("dist - ", dist)
        action = dist.sample().numpy()[0]
        #print("action : ", action)
        return action

class Agent(mp.Process):
    """docstring for Agent"""
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, gamma, lr, name, global_ep_idx, env):
        super(Agent, self).__init__()
        print("INIT in Agent")
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%2i' % name
        self.episode_idx = global_ep_idx
        self.env = env #gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        t_step = 5
        try:
            print("run in Agent")
            while self.episode_idx.value < iterationNum:
                reward = 0
                done = False
                info = None
                observation = env.reset()
                current_state = [observation[8]*0.00001,observation[9]*0.001,observation[11]*0.000001,observation[13]*(1/340)*0.0001,observation[14]*(1/340)*0.0001]
                score = 0
                self.local_actor_critic.clear_memory()
                while not done:
                    print("---obs, reward, done, info: ", current_state, reward, done, info," episode: ",iterationNum)
                    action = self.local_actor_critic.choose_action(current_state)
                    print('CWND change: {}', action)
                    print("Observation ", observation[5])
                    #new_cwnd = observation[5] + action
                    if new_cwnd < 0.0:
                        new_cwnd = 0.0
                    elif new_cwnd > 65535.0:
                        new_cwnd = 65535.0

                    new_ssThresh = observation[4] + 0
                    if new_ssThresh < 0:
                        new_ssThresh = 0
                    elif new_ssThresh > 65535:
                        new_ssThresh = 65535
                    print("new_cwnd",new_cwnd)
                    print("observation 4 : ", observation[4])
                    observation_, reward, done, info = env.step([new_ssThresh, new_cwnd])
                    observation_ = observation
                    score += reward
                    print("rewards : -------------------------- ", reward)
                    next_state = [observation_[8]*0.00001,observation_[9]*0.001,observation_[11]*0.000001,observation_[13]*(1/340)*0.0001,observation_[14]*(1/340)*0.0001]
                    self.local_actor_critic.remember(current_state, action, reward)
                    if t_step % T_MAX == 0 or done:
                        loss = self.local_actor_critic.calc_loss(done)
                        self.optimizer.zero_grad()
                        loss.backward()
                        for local_param, global_param in zip(self.local_actor_critic.parameters(), self.global_actor_critic.parameters()):
                            global_param._grad = local_param._grad
                        self.optimizer.step()
                        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                        self.local_actor_critic.clear_memory()
                    t_step += 1
                    current_state = next_state
                with self.episode_idx.get_lock():
                    self.episode_idx.value += 1
                print(self.name, 'episode ',self.episode_idx.value, 'reward %.1f' % score)
        except KeyboardInterrupt:
            print("Ctrl-C -> Exit")
        finally:
            env.close()



if __name__ == '__main__':
    lr = 1e-4
    parser = argparse.ArgumentParser(description='Start simulation script on/off')
    parser.add_argument('--start',
                        type=int,
                        default=1,
                        help='Start ns-3 simulation script 0/1, Default: 1')
    parser.add_argument('--iterations',
                        type=int,
                        default=1,
                        help='Number of iterations, Default: 1')
    args = parser.parse_args()
    startSim = bool(args.start)
    iterationNum = 100

    port = 0
    simTime = 100 # seconds
    stepTime = 0.1
    seed = 12
    simArgs = {"--duration": simTime,}
    debug = False
    input_dims = [5]
    n_actions = 2
    T_MAX = 5
    env = ns3env.Ns3Env(port=port,startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)

    env.reset()

    ob_space = env.observation_space
    ac_space = env.action_space

    print("Observation space: ", ob_space,  ob_space.dtype)
    print("Action space: ", ac_space, ac_space.dtype)


    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr = lr, betas = (0.92, 0.999))
    global_ep = mp.Value('i', 0)
    
    workers = [Agent(global_actor_critic, optim, input_dims, n_actions, gamma = 0.99, lr = lr, name = i, global_ep_idx = global_ep, env = env) for i in range(mp.cpu_count()) ]
    [w.start() for w in workers]
    [w.join() for w in workers]

#a2c_agent = NStepAgent(alpha=5.19e-5, beta=2.42e-5, input_dims=[5], gamma=0.75,
#                  n_actions=2, layer1_size=25, layer2_size=4)
