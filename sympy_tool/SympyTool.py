#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sympy
import numpy as np
import math
sympy.init_printing(use_latex='mathjax')
from IPython.display import display

S = sympy.sin
C = sympy.cos
#自定義numpy 函式 
np.set_printoptions(precision=3,suppress=True)
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg

class SympyTool(object):
    def __init__(self):
        pass

    def Trans(self,x,y,z):
        trans = sympy.MutableDenseMatrix([[1,0,0,x],
                                        [0,1,0,y],
                                        [0,0,1,z],
                                        [0,0,0,1]])

        return trans
    
    def RotX(self,rx):
        '''
            if rx is value pls input rad
        '''
        rotx = sympy.MutableDenseMatrix([[1,0,0,0],
                                        [0,C(rx),-S(rx),0],
                                        [0,S(rx),C(rx),0],
                                        [0,0,0,1]])

        return rotx
    
    def RotY(self,ry):
        '''
            if ry is value pls input rad
        '''
        roty = sympy.MutableDenseMatrix([[C(ry),0,S(ry),0],
                                        [0,1,0,0],
                                        [-S(ry),0,C(ry),0],
                                        [0,0,0,1]])

        return roty
    
    def RotZ(self,rz):
        '''
            if rz is value pls input rad
        '''
        rotz = sympy.MutableDenseMatrix([[C(rz),-S(rz),0,0],
                                        [S(rz),C(rz),0,0],
                                        [0,0,1,0],
                                        [0,0,0,1]])
        
        return rotz 

    def RotXYZ(self,rx,ry,rz):
        RotX = self.RotX(rx)
        RotY = self.RotY(ry)
        RotZ = self.RotZ(rz)

        return RotX @ RotY @ RotZ

    def Matrix4(self,x,y,z,rx,ry,rz):
        '''
            if rx, ry ,rz is value pls input rad
        '''
        trans = self.Trans(x,y,z)
        rotxyz = self.RotXYZ(rx,ry,rz)

        return trans @ rotxyz

    def RTTR(self,alpha,a,d,theta):
        '''
            if alpha or theta is number pls input rad 
        '''

        transx = self.Trans(a,0,0)
        transz = self.Trans(0,0,d)
        rotx = self.RotX(alpha)
        rotz = self.RotZ(theta)

        return ((rotz @ transz)  @ transx) @ rotx

    def RTTR_Matrix(self,alpha,a,d,theta):
        '''
            if alpha or theta is number pls input rad 
        '''
        T = sympy.MutableDenseMatrix([[C(theta),-S(theta)*C(alpha),S(theta)*S(alpha),a*C(theta)],
                      [S(theta),C(theta)*C(alpha),-C(theta)*S(alpha),a*S(theta)],
                      [0,S(alpha),C(alpha),d],
                      [0,0,0,1]])
        return T    

    def RTRT(self,alpha,a,d,theta):
        '''
            if alpha or theta is number pls input rad 
        '''

        transx = self.Trans(a,0,0)
        transz = self.Trans(0,0,d)
        rotx = self.RotX(alpha)
        rotz = self.RotZ(theta)

        return ((rotx @ transx) @ rotz) @ transz

    def RTRT_Matrix(self,alpha,a,d,theta):
        '''
            if alpha or theta is number pls input rad 
        '''

        T = sympy.MutableDenseMatrix([[C(theta),-S(theta),0,a],
                      [S(theta)*C(alpha),C(theta)*C(alpha),-S(alpha),-d*S(alpha)],
                      [S(theta)*S(alpha),S(alpha)*S(alpha),C(alpha),d*C(alpha)],
                      [0,0,0,1]])
        return T
    
    def Inertia(self,ixx,iyy,izz,ixy,ixz,iyz):
        I = sympy.MutableDenseMatrix([[ixx,ixy,ixz],
                                      [ixy,iyy,izz],
                                      [ixz,iyz,izz]])
        return I


#%%
if __name__ == "__main__":
    tool = SympyTool()
    l1 = sympy.Symbol("l1")
    l2 = sympy.Symbol("l2")

    n = 2
    pi = sympy.pi
    q = sympy.symbols(f'q0:{n}')#input angle

    T1 = tool.RTTR(0,0,0,q[0])
    T2 = tool.RTTR(0,l1,0,q[1])
    T3 = tool.RTTR(0,l2,0,0)

    Ti = [sympy.trigsimp(T1 @ T2),sympy.trigsimp(T1 @ T2 @ T3)]
    display(Ti)
    




