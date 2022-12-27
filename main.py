#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from PPO import PPO
from env import RREnv
from SAC import SAC
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5 import QtCore,QtGui,QtWidgets,QtOpenGL
from PyQt5.QtWidgets import QDialog,QApplication,QMessageBox,QMainWindow,QAction
from RR import RR
from plot import plot_learning_curve,plot_runing_avg_learning_curve
import sys


#自定義numpy 函式 
np.set_printoptions(precision=8,suppress=True)
cos = np.cos
sin = np.sin
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg


################## hyper parameter ####################
app = QApplication(sys.argv)
robot = RR(1,1,0.2,0.2,np.eye(3) * 0.2,np.eye(3) * 0.2)
env = RREnv(robot)
goal_qd = np.zeros(env.state_dimension)
goal = [1,1,4,0,0,0]
env.set_goal(goal,goal_qd)
print("goal",env.goal)
print("goal_q",env.goal_q)
print(env.RR.Fk(env.goal_q)[-1])

n_state = env.state_dimension
n_action = env.action_dimesion

low = RREnv.q_bound[0]
high = RREnv.q_bound[1] 

figure_file = os.path.dirname(__file__)+'/plotPPO/test_v3_action.png'
figure_title = 'Running score of test_RRV4_action'
avg_file = os.path.dirname(__file__)+'/plotPPO/test_RRV4_avg_action.png'
avg_title = 'Running average of test_RRV4_avg_action'

name = '8.pth'
# name = None
save_name = '_v3.pth'

N_game = int(3e6)
A_LR = 3e-4
C_LR = 1e-3
Q_LR = 2e-4
Alpha_LR = 2e-5
sigma = 0.01 
gae_lamda = 0.95
gamma = 0.99
tau = 0.01
sigma_min = 1e-4
sigma_max = 0.2
max_size = 100000
batch_size = 128
EP_LEN = 1000
n_eopc = 80
min_size = 1000
epsloion = 0.2
target_entropy = -n_action
path = os.path.dirname(__file__)+'/SAC/'

save_freq = int(1e5)
update_timestep = EP_LEN * 4
print_freq = EP_LEN * 4
decy_std = int(2e5)
action_std = 0.5
std_rate = 0.05
min_std = 0.1


if __name__ == "__main__":
    best_score = -1000000 
    score_history = []
    # agent = PPO(n_state,n_action,low,high,None,None,std_min=1e-4,std_max=0.5,A_LR=A_LR,C_LR=C_LR,gamma = gamma,gae_lamda = gae_lamda
    #             ,norm_adv=True,epslion=epsloion,n_epoch=n_eopc,batch_size=batch_size,entropy_cofe=0.01
    #             ,fc1=32,fc2=64,fc3=128,fc4=256,fc5=256,save_name=save_name,load_name=name,path=path)
    
    # agent = SAC(n_state,n_action,low,high,None,None,target_entropy,A_LR,C_LR,gamma = gamma,alpha_LR=Alpha_LR,tau=tau,
    #             max_size = max_size,fc1=32,fc2=64,fc3=128,fc4=256,fc5=256,save_name=save_name,load_name=name,path=path,batch_size = batch_size)
    

    
    # time_step = 0
    # i_episode = 0
    
    # # for PPO on-policy method
    # while time_step <= N_game:
    #     s = env.reset()
    #     # done = False
    #     rewards = 0
    #     t0 = time.time()
    #     for t in range(1,EP_LEN+1):
            
    #         a = agent.choose_action(s)
    #         s_,r,done = env.step(a)
            
    #         #store_trainstion
    #         agent.store_transition(s,a,r,s_,done)
    #         # agent.store_transition(s,s_,logprob,a,r,done,val)
    #         # agent.buffer.state.append(s)
    #         # agent.buffer.state_.append(s_)
    #         # agent.buffer.dones.append(done)
    #         # agent.buffer.reward.append(r)
            
    #         rewards += r
    #         time_step += 1
    #         #update PPO
    #         if time_step % update_timestep == 0:
    #             agent.learn()
                
    #         if time_step % print_freq == 0:
    #             print_score = round(avg_score,4)
                
    #             print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_score))
            
    #         # if time_step % decy_std == 0:
    #         #     agent.decay_action_std(std_rate,min_std)
            
    #         if time_step % save_freq == 0:
    #             # if rewards > best_score:
    #                 best_score = rewards
    #                 print("--------------------------------------------------------------------------------------------")
    #                 agent.save_model()
    #                 print("----------------- best_score -------------------- %.3f" % best_score)
                    
    #         if done:
    #             break 
            
    #         s = s_
                    
    #     i_episode += 1          
    #     score_history.append(rewards)
    #     avg_score = np.mean(score_history[-update_timestep:])

    # x = [i+1 for i in range(len(score_history))]
    # plot_learning_curve(x,score_history,figure_title,figure_file)
    # plot_runing_avg_learning_curve(x,score_history,avg_title,avg_file)
    
    
    # for SAC off-policy method
    # for _ in range(N_game):
    #     s = env.reset() 
    #     done = False
    #     start = True
    #     reward = 0
    #     # if (_+1) % 10 == 0:
    #     t0 = time.time()
    #     while not done:
    #         # env.render()
    #         a = agent.choose_action(s)
    #         s_,r,done = env.step(a)
    #         reward += r
    #         agent.store_transition(s,a,r,s_,done)
              
    #         agent.learn()    
    #         # if done:
    #         #     start = False
    #         #     break
    #         s = s_

    #     score_history.append(reward)
    #     avg_score = np.mean(score_history[-100:])
            
    #     if reward > best_score:
    #         best_score = reward
    #         agent.save_model()
    #         print("----------------- best_score -------------------- %.3f" % best_score)
            
    #     # if (_+1) % 10 == 0:
    #     t = time.time() - t0 
    #     print('epoisode each 10 run:',_,'score :%.1f' % reward,'avg score %.1f' % avg_score ,'time %.2f' % t)
            
    # x = [i+1 for i in range(len(score_history))]
    # plot_learning_curve(x,score_history,figure_title,figure_file)
    # plot_runing_avg_learning_curve(x,score_history,avg_title,avg_file)
    
    # test
    # agent.load_model()
    # epoc = 1000
    # while epoc > 0:
    #     s = env.reset()
    #     done = False
    #     while not done:
    #         env.render()
    #         a = agent.choose_action(s)
    #         s_,r,done = env.step(a)
    #         if done:
    #             print(done)
    #         s = s_
