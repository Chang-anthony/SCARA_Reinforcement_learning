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
goal_pid = robot.Matrix4_Q(goal)
# env.set_goal_trajectory(goal_pid,50)
print("goal",env.goal)
print("goal_q",env.goal_q)
print(env.RR.Fk(env.goal_q)[-1])

n_state = env.state_dimension
n_action = env.action_dimesion

low = RREnv.q_bound[0]
high = RREnv.q_bound[1] 

figure_file = os.path.dirname(__file__)+'/plotSAC/test_RRV3_action.png'
figure_title = 'Running score of test_RRV3_action'
avg_file = os.path.dirname(__file__)+'/plotSAC/test_RRV3_avg_action.png'
avg_title = 'Running average of test_RRV3_avg_action'

name = '_v4.pth'
# name = None
save_name = '_v4.pth'

############################ PPO hyper-parameter #####################
# N_game = int(3e6)
# A_LR = 3e-4
# C_LR = 1e-3
# sigma = 0.01 
# gae_lamda = 0.95
# gamma = 0.99
# tau = 0.01
# max_size = 100000
# batch_size = 128
# EP_LEN = 1000
# n_eopc = 80
# min_size = 1000
# epsloion = 0.2
# path = os.path.dirname(__file__)+'/PPO/'


# save_freq = int(1e5)
# update_timestep = EP_LEN * 4
# print_freq = EP_LEN * 4
# decy_std = int(2e5)
# action_std = 0.5
# std_rate = 0.05
# min_std = 0.1

#########################################

#################### SAC parameter #############################
N_game = int(1e6)
A_LR = 1e-4
C_LR = 2e-4
Alpha_LR = 2e-4
sigma = 0.01 
gae_lamda = 0.9
gamma = 0.99
tau = 0.01
max_size = 100000
batch_size = 128
EP_LEN = 3000
n_eopc = 10
min_size = 1000
target_entropy = -n_action
epsloion = 0.2
std_min = 1e-4
std_max = 1
path = os.path.dirname(__file__)+'/SAC/'

save_freq = int(1e5)
update_timestep = EP_LEN 
print_freq = EP_LEN 
decy_std = int(1e5)
action_std = 0.5
std_rate = 0.1
min_std = 0.1

##############################################################

if __name__ == "__main__":
    best_score = -1000000 
    score_history = []
    # agent = PPO(n_state,n_action,low,high,None,None,std_min=1e-4,std_max=0.5,A_LR=A_LR,C_LR=C_LR,gamma = gamma,gae_lamda = gae_lamda
    #             ,norm_adv=True,epslion=epsloion,n_epoch=n_eopc,batch_size=batch_size,entropy_cofe=0.01
    #             ,fc1=32,fc2=64,fc3=128,fc4=256,fc5=256,save_name=save_name,load_name=name,path=path)
    
    agent = SAC(n_state,n_action,low,high,None,None,target_entropy,A_LR,C_LR,std_min=std_min,std_max=std_max
                ,gamma = gamma,alpha_LR=Alpha_LR,tau=tau,max_size=max_size,fc1=32,fc2=64,fc3=128,fc4=256,fc5=256,
                save_name=save_name,load_name=name,path=path,batch_size = batch_size)
    
    
    time_step = 0
    i_episode = 0
    avg_score = 0
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
    
    
    # for SAC DDPG off-policy method
    # while time_step <= N_game:
    #     s = env.reset()
    #     env.set_goal_random()
    #     # done = False
    #     rewards = 0
    #     t0 = time.time()
    #     for t in range(1,EP_LEN+1):
            
    #         a = agent.choose_action(s)
    #         s_,r,done = env.step(a)
            
    #         #store_trainstion
    #         agent.store_transition(s,a,r,s_,done)
            
    #         rewards += r
    #         time_step += 1
    #         #update SAC
    #         if agent.memory.memory_counter >= min_size:
    #             agent.learn()
                
    #         if time_step % print_freq == 0:
    #             print_score = round(avg_score,4)
                
    #             print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_score))
            
    #         if time_step % decy_std == 0:
    #             agent.decay_std_max(std_rate,min_std)
            
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
    

    
    
    
    # test
    agent.load_model()
    epoc = 1000
    while epoc > 0:
        s = env.reset()
        env.set_goal_random()
        
        # t = 0
        # plot_q1 = []
        # plot_q2 = []
        # plot_qd1 = []
        # plot_qd2 = []
        # plot_x = []
        # plot_y = []
        # plot_z = []
        # ts = []
        
        # plot_q1.append(s[0])
        # plot_q2.append(s[1])
        # plot_qd1.append(s[2])
        # plot_qd2.append(s[3])
        # plot_x.append(s[6])
        # plot_y.append(s[7])
        # plot_z.append(s[8])
        # ts.append(t)
        
        done = False
        
        while not done:
            env.render()
            a = agent.choose_action(s)
            s_,r,done = env.step(a)
            if done:
                print(done)
            s = s_
            
            # t += 1
            # plot_q1.append(s[0])
            # plot_q2.append(s[1])
            # plot_qd1.append(s[2])
            # plot_qd2.append(s[3])
            # plot_x.append(s[6])
            # plot_y.append(s[7])
            # plot_z.append(s[8])
            # ts.append(t)
            
        # fig1 = plt.figure(figsize=((12,8)))
        # fig2 = plt.figure(figsize=((12,8)))
        # fig3 = plt.figure(figsize=((12,8)))
        # fig4 = plt.figure(figsize=((12,8)))
        # fig5 = plt.figure(figsize=((12,8)))
        # fig6 = plt.figure(figsize=((12,8)))
        # fig7 = plt.figure(figsize=((12,8)))
        
        # ax = fig1.add_subplot(1,1,1)
        # ax2 = fig2.add_subplot(1,1,1)
        # ax3 = fig3.add_subplot(1,1,1)
        # ax4 = fig4.add_subplot(1,1,1)
        # ax5 = fig5.add_subplot(1,1,1)
        # ax6 = fig6.add_subplot(1,1,1)
        # ax7 = fig7.add_subplot(1,1,1)
        
        # ax.plot(ts,plot_q1,color ='r',label="q1")
        # ax2.plot(ts,plot_q2,color ='r',label="q2")
        # ax3.plot(ts,plot_qd1,color ='r',label="qd1")
        # ax4.plot(ts,plot_qd2,color ='r',label="qd2")
        # ax5.plot(ts,plot_x,color ='r',label="x")
        # ax6.plot(ts,plot_y,color ='r',label="y")
        # ax7.plot(ts,plot_z,color ='r',label="z")
        
        # ax.set_title("SCARA RL q1 curve")
        # ax2.set_title("SCARA RL q2 curve")
        # ax3.set_title("SCARA RL qd1 curve")
        # ax4.set_title("SCARA RL qd2 curve")
        # ax5.set_title("SCARA RL x curve")
        # ax6.set_title("SCARA RL y curve")
        # ax7.set_title("SCARA RL z curve")
        
        # ax.legend()
        # ax2.legend()
        # ax3.legend()
        # ax4.legend()
        # ax5.legend()
        # ax6.legend()
        # ax7.legend()

        # plt.show()
    
    
    # test PID
    # epoc = 1000
    # while epoc > 0:
    #     s = env.reset_pid()
    #     done = False
    #     while not done:
    #         env.render()
    #         done = env.PID_Trajectory()
    #         # a = agent.choose_action(s)
    #         # s_,r,done = env.step(a)
    #         if done:
    #             print(done)
    #         # s = s_
    
    
    # test FreeFall
    epoc = 1000
    while epoc > 0:
        s = env.reset()
        done = False
        while not done:
            env.render()
