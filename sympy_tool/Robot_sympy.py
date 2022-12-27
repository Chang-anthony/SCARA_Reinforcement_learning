#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sympy
import numpy as np
import math
sympy.init_printing(use_latex='mathjax')
from IPython.display import display
from sympy_tool.SympyTool import SympyTool

S = sympy.sin
C = sympy.cos
#自定義numpy 函式 
np.set_printoptions(precision=3,suppress=True)
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg


class Robot_Sympy(SympyTool):
    def __init__(self):
        super(Robot_Sympy,self).__init__()

    def RTTR(self, alpha, a, d, theta):
        return super(Robot_Sympy,self).RTTR(alpha, a, d, theta)
    
    def RTTR_Matrix(self, alpha, a, d, theta):
        return super(Robot_Sympy,self).RTTR_Matrix(alpha, a, d, theta)

    def RTRT(self, alpha, a, d, theta):
        return super(Robot_Sympy,self).RTRT(alpha, a, d, theta)

    def RTRT_Matrix(self, alpha, a, d, theta):
        return super(Robot_Sympy,self).RTRT_Matrix(alpha, a, d, theta)
    
    def Inertia(self, ixx, iyy, izz, ixy, ixz, iyz):
        return super(Robot_Sympy,self).Inertia(ixx, iyy, izz, ixy, ixz, iyz)

    def Jacobian(self,n,q,Ti,rotate_axis):
        '''
            n => number of joints
            Ti => FK list

            return Jacobian matrix 6 by n matrix
        '''
        J = sympy.MutableDenseMatrix(np.zeros((6,n),dtype=np.float32))
        Tend = Ti[-1][:,:]

        for i in range(n):
            J[:3,i] = sympy.Array([sympy.diff(Tend[0,3],q[i]).trigsimp(),sympy.diff(Tend[1,3],q[i]).trigsimp(),sympy.diff(Tend[2,3],q[i]).trigsimp()])
            if rotate_axis[i] == 'z':
                J[3:6,i] = Ti[i][:3,2]
            elif rotate_axis[i] == '-z':
                J[3:6,i] = -Ti[i][:3,1]
            elif rotate_axis[i] == 'y':
                J[3:6,i] = Ti[i][:3,1]
            elif rotate_axis[i] == '-y':
                J[3:6,i] = -Ti[i][:3,1]
            elif rotate_axis[i] == 'x':
                J[3:6,i] = Ti[i][:3,0]
            elif rotate_axis[i] == '-x':
                J[3:6,i] = -Ti[i][:3,0]

        return J
        
    def Jacobian_Matrix(self,n,q,Tend):
        '''
            n => number of joints
            Tend => FK end is srcTend transform matrix

            return Jacobian matrix 12 by n matrix
        '''
        J = sympy.MutableDenseMatrix(np.zeros((12,n),dtype=np.float32))
        for i in range(n):
            #x
            J[:3,i] = sympy.Array([sympy.diff(Tend[0,0],q[i]).trigsimp(),sympy.diff(Tend[1,0],q[i]).trigsimp(),sympy.diff(Tend[2,0],q[i]).trigsimp()])
            #y
            J[3:6,i] = sympy.Array([sympy.diff(Tend[0,1],q[i]).trigsimp(),sympy.diff(Tend[1,1],q[i]).trigsimp(),sympy.diff(Tend[2,1],q[i]).trigsimp()])
            #z
            J[6:9,i] = sympy.Array([sympy.diff(Tend[0,2],q[i]).trigsimp(),sympy.diff(Tend[1,2],q[i]).trigsimp(),sympy.diff(Tend[2,2],q[i]).trigsimp()])
            #V
            J[9:12,i] = sympy.Array([sympy.diff(Tend[0,3],q[i]).trigsimp(),sympy.diff(Tend[1,3],q[i]).trigsimp(),sympy.diff(Tend[2,3],q[i]).trigsimp()])

        return J

    def Jacobian_geometry(self,n,Ti,rotate_axis):
        '''
            return 12 by n Jacobian
        '''
        J = sympy.MutableDenseMatrix(np.zeros((12,n),dtype=np.float32))
        Tend = Ti[-1]
        for i in range(n):
            if rotate_axis[i] == 'z':
                zn = Ti[i][:3,2]
            elif rotate_axis[i] == '-z':
                zn = -Ti[i][:3,2]
            elif rotate_axis[i] == 'y':
                zn = Ti[i][:3,1]
            elif rotate_axis[i] == '-y':
                zn = -Ti[i][:3,1]
            elif rotate_axis[i] == 'x':
                zn = Ti[i][:3,0]
            elif rotate_axis[i] == '-x':
                zn = -Ti[i][:3:0]
            
            rn = Tend[:3,3] - Ti[i][:3,3]

            J[:3,i] = sympy.trigsimp(sympy.Array(np.cross(zn,Tend[:3,0])))
            J[3:6,i] = sympy.trigsimp(sympy.Array(np.cross(zn,Tend[:3,1])))
            J[6:9,i] = sympy.trigsimp(sympy.Array(np.cross(zn,Tend[:3,2])))
            J[9:12,i] = sympy.trigsimp(sympy.Array(np.cross(zn,rn)))

        return J

    def Lagarange(self,n,q,qd,Ti,rotate_axis,mass,Pc,G_axis='z',Inertia = None):
        '''
            q,qd,qdd is always from 1 to n+1
            n => number of joints
            Ti => FK list
            Pc => Center mass postion list
            Inertia => Inertia Matrix list if None is ignore
            rotate_axis => joints default ratate frame 
            G_axis => gravity affect axis input str default is 'z'


            return M,V,C,G,Jv,Jw,J,KE,Ki,PE
        '''
        # q = sympy.symbols(f'q1:{n+1}')#input angle
        # qd = sympy.symbols(f'qd1:{n+1}')#input angle vel
        # qdd = sympy.symbols(f'qdd1:{n+1}')#input angle acc
        g = sympy.Symbol('g')

        #angular velocity jacobian (Jw)
        Jw = sympy.MutableDenseNDimArray(np.zeros((n,3,n),dtype = np.int32))
        for i in range(n):
            for j in range(i+1):
                if rotate_axis[j] == 'z':
                    Jw[i,:,j] = Ti[j][:3,2]
                elif rotate_axis[j] == '-z':
                    Jw[i,:,j] = -Ti[j][:3,2]
                elif rotate_axis[j] == 'y':
                    Jw[i,:,j] = Ti[j][:3,1]
                elif rotate_axis[j] == '-y':
                    Jw[i,:,j] = -Ti[j][:3,1]
                elif rotate_axis[j] == 'x':
                    Jw[i,:,j] = Ti[j][:3,0]
                elif rotate_axis[j] == '-x':
                    Jw[i,:,j] = -Ti[j][:3,0]

        #linear velocity jacobian (Jv)
        Jv = sympy.MutableDenseNDimArray(np.zeros((n,3,n),dtype = np.int32))
        for i in range(n):
            P = Ti[i][:,:] @ Pc[i][:,:]

            cx = P[0,3]
            cy = P[1,3]
            cz = P[2,3]

            for j in range(n):
                Jv[i,:,j] = sympy.Array([sympy.diff(cx,q[j]).simplify(),sympy.diff(cy,q[j]).simplify(),sympy.diff(cz,q[j]).simplify()])

        
        J = sympy.Matrix(np.vstack((Jv[-1,:,:],Jw[-1,:,:])))
        
        #Potential energy and Kinetic energy and Mass Matrix M
        P = sympy.MutableDenseMatrix(np.eye(4,dtype=np.float32))
        PE = 0
        #M is Mass Matrix  Ki is Kinetic energy for each joint
        M = sympy.MutableDenseMatrix(np.zeros((n,n),dtype=np.float32))
        Ki = []
        '''
        Kinetic energy 
            Ki = mass[i] * Jv[i]' * Jv[i] + Jw[i]' * Inertia[i] * Jw[i]
            MassMatrix = sum of Ki 
            KE = 0.5 * (qd' * MassMatrix * qd)
        Potential energy
            PE = sum of (mass[i] * g * h)     
        '''
        for i in range(n):
            P = Ti[i][:,:] @ Pc[i][:,:]
            if G_axis == 'z': 
                PE = PE + mass[i] * g * P[2,3]
            elif G_axis == 'y':
                PE = PE + mass[i] * g * P[1,3]
            elif G_axis == 'x':
                PE = PE + mass[i] * g * P[0,3]
            
            if Inertia is not None:
                ki = sympy.trigsimp(mass[i] * sympy.MutableDenseMatrix(np.dot(Jv[i].transpose(),Jv[i])) + sympy.MutableDenseMatrix(Jw[i].transpose() @ Inertia[i] @ Jw[i]))
            else :
                ki = sympy.trigsimp(mass[i] * sympy.MutableDenseMatrix(np.dot(Jv[i].transpose(),Jv[i])))
            M = M + ki
            Ki.append(ki)
        
        KE = 0.5 * sympy.MutableDenseMatrix(qd).transpose() @ M @ sympy.MutableDenseMatrix(qd)
        
        #Gravity Matrix 
        G = sympy.MutableDenseMatrix(np.zeros((n,1),dtype=np.float32))

        for i in range(n):
            G[i] = sympy.diff(PE,q[i]).trigsimp()

        #Christoffel symbol Matrix
        c = sympy.MutableDenseNDimArray(np.zeros((n,n,n),dtype=np.float32))
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    c[i,j,k] = 0.5 * (sympy.diff(M[k,j],q[i]) + sympy.diff(M[k,i],q[j]) - sympy.diff(M[i,j],q[k]))


        C = sympy.MutableDenseMatrix(np.zeros((n,n),dtype=np.float32))

        for k in range(n):
            for j in range(n):
                temp = 0
                for i in range(n):
                    temp = temp + c[i,j,k] * qd[i]
                C[k,j] = sympy.trigsimp(temp)


        V = sympy.MutableDenseMatrix(np.dot(C,qd))

        return M,V,C,G,Jv,Jw,J,KE,Ki,PE    

    def Euler_Lagarange(self):
        pass        

