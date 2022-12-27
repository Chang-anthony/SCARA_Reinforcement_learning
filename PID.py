#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony
"""
import numpy as np
import math

class PID(object):
    def __init__(self,KU = 1,KP = 1,KI = 0,KD = 0.1,TU = None,action_bound = [-1,1]):
        self.KU = KU
        self.TU = TU
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.Bound = action_bound
        self.I = 0
    
    
    def PID_ZN(self,KU,TU):
        self.KU = KU
        self.TU = TU
        self.KP = 0.6 * KU
        self.KI = 2 * self.KP / self.TU
        self.KD = self.KP * TU / 8
        
    def Pessen_Integral_ZN(self,KU,TU):
        self.KU = KU
        self.TU = TU
        self.KP = 0.7 * KU
        self.KI = 2.5 * self.KP / self.TU
        self.KD = 0.15 * self.KP * self.TU 
        
    def Simple_overshoot(self,KU,TU):
        self.KU = KU 
        self.TU = TU
        self.KP = 0.33 * KU
        self.KI = 2 * self.KP / TU
        self.KD = self.KP * TU /3 
    
    def Update(self,error,derror,dt_Pid):
        
        P = error * self.KP
        self.I = self.I + error * self.KI * dt_Pid
        D = derror * self.KD / dt_Pid
        
        Out = P + self.I + D
        if self.Bound != None:
            Out = np.clip(Out,self.Bound[0],self.Bound[1])
        
        return Out
    
    def Clip(self,Input):
        if self.Bound != None:
            Input = np.clip(Input,self.Bound[0],self.Bound[1])
        
        return Input
    