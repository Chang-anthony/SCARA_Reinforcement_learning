#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
from re import L
import re
from turtle import tracer
from typing import overload 
import numpy as np
import matplotlib.pyplot as plt
from numpy import core
from numpy.core.fromnumeric import shape, std
from numpy.lib.function_base import select

class  Matrix4:

    def __init__(self):
        self.coord = np.eye(4,dtype='float')
        
    def Create_Q(self,x,y,z,rx,ry,rz):
        trans = self.Trans(x,y,z)
        rotxyz = self.RotXYZ(rx,ry,rz)
        
        out = trans * rotxyz
        
        return out
    
    def Create_Qlist(self,Q):
        trans = self.Trans(Q[0],Q[1],Q[2])
        rotxyz = self.RotXYZ(Q[3],Q[4],Q[5])
        #out = trans.__mul__(rotxyz)
        out = trans * rotxyz  
                 
        return out
    
    def Create_Crood(self,coord):
        out = Matrix4()
        out.coord = coord
        
        return out    
    
    def Trans(self,x,y,z):
        out = Matrix4()
        trans = Matrix4()
        trans.coord[0,3] = float(x)
        trans.coord[1,3] = float(y)
        trans.coord[2,3] = float(z)
        
        #out = self.__mul__(trans)
        out = out * trans 
        
        return out
         
    def TransX(self,dx):
        transx = self.Trans(dx,0,0)
        
        return  transx
    
    def TransY(self,dy):
        transy = self.Trans(0,dy,0)
        
        return  transy
    
    def TransZ(self,dz):
        transz = self.Trans(0,0,dz)
        
        return  transz
        
    def Rotx(self,deg):
        dst = Matrix4()
        rotx = Matrix4()
        rad = deg * math.pi / 180
        
        rotx.coord[0][0] = 1
        rotx.coord[3][3] = 1
        rotx.coord[1][1] = math.cos(rad)
        rotx.coord[1][2] = -math.sin(rad)
        rotx.coord[2][1] = math.sin(rad)
        rotx.coord[2][2] = math.cos(rad)
        
        dst = dst * rotx
        
        return dst
    
    def Roty(self,deg):
        # 右手座標係
        dst = Matrix4()
        roty = Matrix4()
        rad = deg * math.pi / 180
        roty.coord[1][1] = 1
        roty.coord[3][3] = 1
        roty.coord[0][0] = math.cos(rad)
        roty.coord[0][2] = math.sin(rad)
        roty.coord[2][0] = -math.sin(rad)
        roty.coord[2][2] = math.cos(rad)
        
        dst = dst * roty
        return dst
    
    def Rotz(self,deg):
        dst = Matrix4()
        rad = deg * math.pi / 180
        rotz = Matrix4()
        rotz.coord[2][2] = 1
        rotz.coord[3][3] = 1
        rotz.coord[0][0] = math.cos(rad)
        rotz.coord[0][1] = -math.sin(rad)
        rotz.coord[1][0] = math.sin(rad)
        rotz.coord[1][1] = math.cos(rad)
        
        dst = dst * rotz
        
        return dst
    
    def RotXYZ(self,rx,ry,rz):
        matxyz = Matrix4()
        matx = self.Rotx(rx)
        maty = self.Roty(ry)
        matz = self.Rotz(rz)
        
        matxyz = matz * maty * matx 
        
        return matxyz
    
    
    def Get_Rot(self):
        return np.array(self.coord[0:3,0:3])
    
    def Get_Trans(self):
        return np.array(self.coord[0:3,3])
    
    def getcurrent_Angle(self):

        r11 = self.coord[0,0]#Xx
        r21 = self.coord[1,0]#Xy
        r31 = self.coord[2,0]#Xz

        r12 = self.coord[0,1]#Yx
        r22 = self.coord[1,1]#Yy

        r32 = self.coord[2,1]#Yz
        r33 = self.coord[2,2]#ZZ

        β = math.atan2(-r31,math.sqrt(r11 * r11+r21 * r21))
        alpha = 0
        gamma = 0
        if abs(β) != 90:
            alpha = math.atan2(r21/math.cos(β),r11/math.cos(β))
            gamma = math.atan2(r32/math.cos(β),r33/math.cos(β))
        else:
            if β == 90:
                alpha = 0
                gamma = math.atan2(r12,r22)  
            elif  β == -90:
                alpha = 0
                gamma = -math.atan2(r12,r22) 
        
        return [gamma *180 / math.pi ,β *180 / math.pi,alpha *180 / math.pi]
        
    def getcurrent_TransandAngle(self):
        q = np.zeros(shape=(6))
        r11 = self.coord[0,0]#Xx
        r21 = self.coord[1,0]#Xy
        r31 = self.coord[2,0]#Xz

        r12 = self.coord[0,1]#Yx
        r22 = self.coord[1,1]#Yy

        r32 = self.coord[2,1]#Yz
        r33 = self.coord[2,2]#ZZ

        β = math.atan2(-r31,math.sqrt(r11 * r11+r21 * r21))
        alpha = 0
        gamma = 0
        if abs(β) != 90:
            alpha = math.atan2(r21/math.cos(β),r11/math.cos(β))
            gamma = math.atan2(r32/math.cos(β),r33/math.cos(β))
        else:
            if β == 90:
                alpha = 0
                gamma = math.atan2(r12,r22)  
            elif  β == -90:
                alpha = 0
                gamma = -math.atan2(r12,r22) 
    
        q = [self.coord[0,3], self.coord[1,3], self.coord[2,3], gamma *180 / math.pi ,β *180 / math.pi,alpha *180 / math.pi ]
        return q
    
    
    def Sclar(self,sclar):
        dst = Matrix4()
        dst.coord = self.coord * sclar
        return dst
        
    # 重載運算子 *  
    def __mul__(self,other): 
        dst = Matrix4()
        
        if isinstance(other,Matrix4):
            dst.coord = self.coord @ other.coord
            return dst
        else:
            tmp = self.Create_Crood(other)
            dst.coord = self.coord @ tmp.coord
            return dst
      

           
    # 重載運算子 -
    def __sub__(self,other):
        dst = Matrix4()
        
        if isinstance(other,Matrix4):
            dst.coord = self.coord - other.coord
            return dst
        else:
            tmp = self.Create_Crood(other)
            dst.coord = self.coord - tmp.coord
            return dst
        
        
    # 重載運算子 + 
    def __add__(self,other):
        dst = Matrix4()
        
        if isinstance(other,Matrix4):
            dst.coord = self.coord + other.coord
        else:
            tmp = self.Create_Crood(other)
            dst.coord = self.coord + tmp.coord
            return dst
    
    def inverse(self):
        inv = Matrix4()
        inv.coord = np.linalg.inv(self.coord)
        return inv
    
    #自己移動 
    def Move(self,dx,dy,dz):
        self.coord[0,3] += dx
        self.coord[1,3] += dy
        self.coord[2,3] += dz
       
    #R(T) + (RT)(3*3)*(P)(3*1)*(-1)
    def inverse_Tmat(self):
        inv = Matrix4()
        
        for i in range(3):
            for j in range(3):
                inv.coord[i][j] = self.coord[j][i]
        
        for i in range(3):
            inv.coord[0][3] = 0
            for j in range(3):
                inv.coord[i][3] += inv.coord[i][j] * self.coord[j][3]
            inv.coord[i][3] = -inv.coord[i][3]
        inv.coord[3][3] = 1
        
        return inv 

class Point3D:

    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def dot(self,a,b):
        if isinstance(a,Point3D) and isinstance(b,Point3D):
            return a.x*b.x+a.y*b.y+a.z*b.z
        else:
            print("a and b is not 3D vector")

    def cross(self,a,b):

        if isinstance(a,Point3D) and isinstance(b,Point3D):
            c = Point3D(0,0,0)
            c.x = (a.y*b.z-a.z*b.y)
            c.y = -(a.x*b.z-a.z*b.x)
            c.z = (a.x*b.y-a.y*b.x)

            return c
        else:
            print("a and b is not 3D vector")

    #單位向量
    def norm(self,a):
        if isinstance(a,Point3D):
            len = math.sqrt(a.x*a.x+a.y*a.x+a.z*a.z)
            c = Point3D(a.x/len,a.y/len,a.z/len)
            return c
        else:
            print("a is not a 3D vector")      
                                


     







    
    