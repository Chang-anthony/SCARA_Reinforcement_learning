#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from turtle import done
import numpy as np
import collections 
import random


class DataBuffer:
    def __init__(self):
        self.buffer = []
    
    def store_transition(self,state,action,reward,new_state,done):
        self.buffer.append((state,action,reward,new_state,done))
        
        
    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.buffer:
            s, a, r, s_, done = item
            l_s.append(s)
            l_a.append(a)
            l_r.append(r)
            l_s_.append(s_)
            l_done.append(done)
        return l_s, l_a, l_r, l_s_, l_done
    
    def clear(self):
        self.buffer.clear()

class My_ReplayBuffer:
    def __init__(self,max_size):
        self.buffer = collections.deque(maxlen=max_size)
    
    def store_transition(self,state,action,reward,new_state,done):
        self.buffer.append((state,action,reward,new_state,done))
        
    def sample(self, batch_size): 
        '''
            return state , action ,reward , state_ ,done
        '''
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done 
    
    def sample_all(self):
        transitions = random.sample(self.buffer,self.size())
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done 
    
    def sample_no_randonm(self):
        state, action, reward, next_state, done = zip(*self.buffer)
        
        return state, action, reward, next_state, done 
    
    def clear(self):
        self.buffer.clear()
    
    def size(self): 
        return len(self.buffer)
    



class ReplayBuffer:
    def __init__(self,max_size,input_shape,n_actions):
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size,input_shape))
        self.new_state_memory = np.zeros((self.memory_size,input_shape))
        self.action_memory = np.zeros((self.memory_size,n_actions))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size,dtype=np.bool)
    
    def store_transition(self,state,action,reward,new_state,done):
        index = self.memory_counter % self.memory_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self,batch_size):
        '''
        states,actions,rewards,states_,dones
        '''
        max_mem = min(self.memory_counter,self.memory_size)

        batch = np.random.choice(max_mem,batch_size,replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states,actions,rewards,states_,dones