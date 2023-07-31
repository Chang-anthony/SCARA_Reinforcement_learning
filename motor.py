#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony
"""
#%%
import numpy as np
import copy
import math
import matplotlib.pyplot as plt


class Motor(object):
    def __init__(self):
        self.P = 885  
        self.I = 0 
        self.D = 0
        
        self.q_buffer = []
        self.qd_buffer = []
        self.goalq_buffer = []
        self.goalqd_buffer = []
        self.torque_buffer = []
        self.load_buffer = []
        self.errq_buffer = []
        self.errqd_buffer = []
        self.coputeI = 0
    
    def reset(self,P,I,D,q_intial,qd_inital):
        self.P = P
        self.I = I 
        self.D = D
        
        self.q_buffer = []
        self.qd_buffer = []
        self.goalq_buffer = []
        self.goalqd_buffer = []
        self.torque_buffer = []
        self.load_buffer = []
        self.errq_buffer = []
        self.errqd_buffer = []
        
        self.coputeI = 0
    
    def update(self,goal_q,goal_qd,q,qd,Fd):
        '''
            Fd dynamics eqution load torque
        '''
        
        self.goalq_buffer.append(goal_q)
        self.goalqd_buffer.append(goal_qd)
        self.q_buffer.append(q)
        self.qd_buffer.append(qd)
        self.load_buffer.append(Fd)
    
        error_q = goal_q - self.q_buffer[-1]
        error_qd = goal_qd - self.qd_buffer[-1]
        
        self.errq_buffer.append(error_q)
        self.errqd_buffer.append(error_qd)
        
        self.coputeI += error_q * self.I
        pwm = error_q * self.P + self.coputeI + self.D * error_qd
        #16384*3.14/18 = pwm = 100% errorq = 10deg errorq = 10*3.14/180
        pwm /= 16384 * np.deg2rad(1)
        pwm = np.clip(pwm,-1,1)
        
        efficency = Motor.Performacnce_Regreesion(Fd)
        power = pwm * efficency * 8
        self.torque_buffer.append(power)
        
        return power
        
        
        #12V 41rpm
        #6V  41/2rpm
        #
    @staticmethod   
    def Performacnce_Regreesion(Fd):
        '''
            mx_106 efficency line Regreesion output efficency
            https://emanual.robotis.com/docs/en/dxl/mx/mx-106/
        '''
        
        Fd = np.clip(abs(Fd),0,5.4)
        efficency = 0.0023 * Fd ** 3 - 0.0576 * Fd ** 2 + 0.2267 * Fd + 0.2374
        
        return efficency
    
    def OverLoadError(self,weight = 0.2):
        '''
            load torque error
        '''
        return weight * np.sum(np.array(self.torque_buffer) - np.array(self.load_buffer))
    
    
    def Get_buffers(self):
        '''
            return buffer datas
            q,qd,goalq,goalqd,errq,errqd,load,output
        '''
            
        return self.q_buffer,self.qd_buffer,self.goalq_buffer,self.goalqd_buffer,\
            self.errq_buffer,self.errqd_buffer,self.load_buffer,self.torque_buffer
        
        
        
        