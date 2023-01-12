#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
from plot import plot_learning_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torch.backends import cudnn
from buffer import DataBuffer,My_ReplayBuffer,ReplayBuffer
path = os.path.dirname(__file__)+'/'


# torch.cuda.set_device(0) #import  part
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self,n_state,n_action,qhigh,qdhigh=None ,LR=None,std_min=1e-4,std_max=1,fc1=64,fc2=128,fc3=256,fc4=256,fc5=256):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(n_state,fc1).to(device)
        self.fc2 = nn.Linear(fc1,fc2).to(device)
        self.fc3 = nn.Linear(fc2,fc3).to(device)
        self.fc4 = nn.Linear(fc3,fc4).to(device)
        self.fc5 = nn.Linear(fc4,fc5).to(device)
        
        self.mu = nn.Linear(fc5,n_action).to(device)
        self.std = nn.Linear(fc5,n_action).to(device)
        
        self.n_state = n_state
        self.n_action = n_action
        self.std_min = std_min
        self.std_max = std_max
        
        self.qhigh = qhigh
        self.qdhigh = qdhigh 
        self.reparm_noise = 1e-4
        
        if LR is not None:
            self.policy_optim = torch.optim.Adam(self.parameters(),lr=LR)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        
        mu = self.mu(x)
        std = F.softplus(self.std(x))
        std = torch.clamp(std,min=self.std_min,max=self.std_max)
        dist = torch.distributions.Normal(mu,std)
        
        normal_sample = dist.rsample() #重參數採樣
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        
        if self.qdhigh is not None:
            actionq = action[:,0:2] * self.qhigh
            actionqd = action[:,2:self.n_action] * self.qdhigh
            action = torch.cat([actionq,actionqd],dim=1) 
        else:
            action = action * self.qhigh
        
        #計算tanh_normal 分佈的對數概率密度
        log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + self.std_min)
        log_prob = torch.sum(log_prob,dim=1,keepdim=True)
        # action = action * self.high

        return action,log_prob
    
    def Set_std(self,std_max):
        self.std_max = std_max


class QValue(torch.nn.Module):
    def __init__(self, n_state,n_action,LR=None,fc1=64,fc2=128,fc3=256):
        super(QValue,self).__init__()
        self.fc1 = torch.nn.Linear(n_state + n_action, fc1).to(device)
        self.fc2 = torch.nn.Linear(fc1, fc2).to(device)
        self.fc3 = nn.Linear(fc2,fc3).to(device)
        
        self.v = torch.nn.Linear(fc3, 1).to(device)
        if LR is not None:
            self.v_optim = torch.optim.Adam(self.parameters(),lr=LR)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return self.v(x)

class SAC(object):
    
    def __init__(self,
                 n_state,
                 n_action,
                 action_q_low,
                 action_q_high,
                 action_qd_low,
                 action_qd_high,
                 target_entorpy, 
                 A_LR = 1e-4,
                 C_LR = 2e-4,
                 alpha_LR = 1e-4,
                 tau = 5e-3,
                 gamma = 0.99,
                 std_min = 1e-4,
                 std_max = 1,
                 fc1 = 64,
                 fc2 = 128,
                 fc3 = 256,
                 fc4 = 256,
                 fc5 = 256,
                 max_size = 1e6,
                 batch_size = 64,
                 load_name = '.pth',
                 save_name = 'v1.pth',
                 path = os.path.dirname(__file__)+'/SAC/',
                 ):
        
        self.actor = Policy(n_state,n_action,action_q_high,action_qd_high,LR=A_LR,std_min=std_min,std_max=std_max,fc1=fc1,fc2=fc2,fc3=fc3,fc4=fc4,fc5=fc5).to(device)
        
        self.critic1 = QValue(n_state,n_action,LR=C_LR,fc1=fc1,fc2=fc2,fc3=fc3).to(device)
        
        self.critic2 = QValue(n_state,n_action,LR=C_LR,fc1=fc1,fc2=fc2,fc3=fc3).to(device)
        
        self.target_critic1 = QValue(n_state,n_action,fc1=fc1,fc2=fc2,fc3=fc3).to(device)
        
        self.target_critic2 = QValue(n_state,n_action,fc1=fc1,fc2=fc2,fc3=fc3).to(device)


        self.n_state = n_state
        self.n_action = n_action
        self.std_min = std_min
        self.std_max = std_max
        self.tau = tau
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.alpha_LR = alpha_LR
        self.target_entorpy = target_entorpy
        self.gamma = gamma
        self.batch_size = batch_size
        self.load_name = load_name
        self.save_name = save_name
        self.path = path
        self.qlow = action_q_low
        self.qhigh = action_q_high
        self.qdlow = action_qd_low
        self.qdhigh = action_qd_high
        self.max_size = max_size
        
        self.log_alpha = torch.tensor(np.log(0.01),dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optim = torch.optim.Adam([self.log_alpha],lr=alpha_LR)
        self.memory = ReplayBuffer(max_size,n_state,n_action)
        # self.memory = My_ReplayBuffer(max_size)
        self.buffer = DataBuffer()
        self.load_model()
    
    def __soft_update(self,old_net,target_net):
        for  target,net in zip(target_net.parameters(),old_net.parameters()):
            target.data.copy_(target.data * (1-self.tau) + net.data * self.tau)
    
    def decay_std_max(self,std_rate,min_std):
        if self.std_max >= min_std:
            self.std_max -= std_rate
            self.actor.Set_std(self.std_max)
        else:
            self.std_max = min_std
            self.actor.Set_std(self.std_max)
        
        print("set std_max to:",self.std_max)
    
    
    def store_transition(self,s,a,r,s_,done):
        # self.memory.store_transition(s,a,r,s_,done)
        self.memory.store_transition(s,a,r,s_,done)
        # self.buffer.store_transition(s,a,r,s_,done)
    
    def choose_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(device)
        with torch.no_grad():
            action = self.actor(state)[0]
            action = action.cpu().detach().numpy()
            action = action.reshape(-1)
        
        return action
    

    def calc_tdtarget(self,rewards,state_,dones):
        next_actions, log_prob = self.actor(state_)
        entropy = -log_prob
        q1_value = self.target_critic1(state_, next_actions)
        q2_value = self.target_critic2(state_, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        
        return td_target
    
    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return 
        s,a,r,s_,done = self.memory.sample_buffer(self.batch_size)
        
        # if self.memory.size() < self.batch_size:
        #     return 
        # s,a,r,s_,done = self.memory.sample(self.batch_size)
        # s,a,r,s_,done = self.memory.sample(self.batch_size)
        # if len(r) < self.batch_size:
        #     return
    
        state = torch.tensor(s,dtype=torch.float32).to(device)
        action = torch.tensor(a,dtype=torch.float32).reshape(-1,self.n_action).to(device)
        reward = torch.tensor(r,dtype=torch.float32).reshape(-1,1).to(device)
        next_state = torch.tensor(s_,dtype=torch.float32).to(device)
        dones = torch.tensor(done,dtype=torch.float32).reshape(-1,1).to(device)

        
        # 更新两个Q网络
        td_target = self.calc_tdtarget(reward, next_state, dones)
        v1 = self.critic1(state, action)
        v2 = self.critic2(state, action)
        critic_1_loss = torch.mean(F.mse_loss(v1, td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(v2, td_target.detach()))
        
        #更新價值網路
        self.critic1.v_optim.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(),0.2)
        self.critic1.v_optim.step()
        self.critic2.v_optim.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(),0.2)
        self.critic2.v_optim.step()
        
        # 更新策略网络
        new_actions, log_prob = self.actor(state)
        entropy = -log_prob
        q1_value = self.critic1(state, new_actions)
        q2_value = self.critic2(state, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor.policy_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.2)
        self.actor.policy_optim.step()
        
        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entorpy).detach() * self.log_alpha.exp())
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

        self.__soft_update(self.critic1, self.target_critic1)
        self.__soft_update(self.critic2, self.target_critic2)
    
    
    def save_model(self):
        torch.save(self.actor.state_dict(),self.path+'actor'+self.save_name)
        torch.save(self.critic1.state_dict(),self.path+'critic1'+self.save_name)
        torch.save(self.critic2.state_dict(),self.path+'critic2'+self.save_name)
        torch.save(self.target_critic1.state_dict(),self.path+'target_critic1'+self.save_name)
        torch.save(self.target_critic2.state_dict(),self.path+'target_critic2'+self.save_name)
        print("-------------------- save model -----------------")
            
    def load_model(self):
        try:
            self.actor.load_state_dict(torch.load(self.path+'actor'+self.load_name))
            self.critic1.load_state_dict(torch.load(self.path+'critic1'+self.load_name))
            self.critic2.load_state_dict(torch.load(self.path+'critic2'+self.load_name))
            self.target_critic1.load_state_dict(torch.load(self.path+'target_critic1'+self.load_name))
            self.target_critic2.load_state_dict(torch.load(self.path+'target_critic2'+self.load_name))
            print("--------------------- load model --------------------")
        except:
            print('-------------------------- load fail ------------------')
            pass


################## hyper parameter ###########################
env_name = 'Pendulum-v1'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
low = env.action_space.low[0]
high = env.action_space.high[0]  # 动作最大值
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

actor_lr = 1e-4
critic_lr = 3e-4
alpha_lr = 3e-4
num_episodes = 1000
hidden_dim = 128
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 1000
batch_size = 64
min_size = 1000
target_entorpy = -env.action_space.shape[0]
path = os.path.dirname(__file__)

figure_file = os.path.dirname(__file__)+'/plots/test1.png'
figure_title = 'Running average of test'
# name = '.pth'
name = None
save_name = '1.pth'


if __name__ == "__main__":
    best_score = -1000000 
    score_history = []
    avg_score = 0
    
    agent = SAC(state_dim,action_dim,low,high,target_entorpy,A_LR=actor_lr,C_LR=critic_lr,
                max_size=buffer_size,alpha_LR=alpha_lr,tau=tau,gamma=gamma,load_name=name,save_name=save_name,path=path)
    
    
    for _ in range(num_episodes):
        s = env.reset()
        done = False
        reward = 0
        t0 = time.time()
        
        while not done:
            #env.render()
            a = agent.choose_aciton(s)
            s_,r,done,info = env.step(a)
            reward += r
            agent.store_transition(s,a,r,s_,done)
            
            # if agent.memory.size() > min_size:
            agent.learn()
            s = s_
        
        score_history.append(reward)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()
        
        if (_+1) % 10 == 0:
            t = time.time() - t0 
            print('epoisode each 10 run:',_,'score :%.1f' % reward,'avg score %.1f' % avg_score ,'time %.2f' % t)
        
        
    x = [i+1 for i in range(num_episodes)]
    plot_learning_curve(x,score_history,figure_file)