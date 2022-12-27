#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony
"""
#%%
import numpy as np
import copy
import math
import sympy
from sympy import lambdify
from sympy_tool.Robot_sympy import Robot_Sympy
sympy.init_printing(use_latex='mathjax')
import matplotlib.pyplot as plt


#自定義numpy 函式 
# np.set_printoptions(precision=3,suppress=True)
cos = np.cos
sin = np.sin
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg
eye = np.eye

class RR(object):
    def __init__(self,L1 = 10,L2 = 10,m1 = 0.5 , m2 = 0.5 , I1 = eye(3) ,I2 = eye(3)):
        #self.dt = 0.002 #1ms
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.I1 = I1
        self.I2 = I2
        self.firction = 0.01

        self.src = np.eye(4)
        self.p1 = np.eye(4)
        self.end = np.eye(4)
        
        self.degs = np.zeros(2)
        self.bufferdegs = np.zeros(2)
        self.ResetPos()
        self.__Create_Dynamics_symbol()

    def Init(self):
        deg = np.zeros(2)
        rad = degTrad(deg)
        src_pos = np.array([0,0,4])
        self.src[:3,3] = src_pos
        Pos = self.Fk(rad)

        return Pos,deg

    def Init_Pos(self):
        self.src = np.eye(4)
        Pos,deg = self.Init()
        self.bufferdegs = deg

        self.Fixed_Src(Pos)

        return Pos,deg

    def ResetPos(self):
        '''
            return degs Pos  
        '''
        self.src = np.eye(4)
        src_pos = np.array([0,0,4])
        self.src[:3,3] = src_pos

        Pos,rad = self.Workready()
        self.bufferdegs = rad
        self.Fixed_Src(Pos)

        return Pos,rad

    def Fixed_Src(self,Pos):
        self.src = self.src @ Pos[0]
        self.p1 = self.src @ Pos[1]
        self.end = self.src @ Pos[2]
    
    def Workready(self):
        deg = np.zeros(2)

        deg[0] = 10
        deg[1] = 10
        rad = degTrad(deg)

        Pos = self.Fk(rad)

        return Pos,rad

    def TransXYZ(self,x = 0,y= 0 ,z = 0):
        trans = np.eye(4)

        trans[0,3] = x
        trans[1,3] = y
        trans[2,3] = z

        return trans
    
    def RotX(self,rad):
        '''
            input rad 
        '''
        #rad = degTrad(deg)
        rotx = np.eye(4)

        rotx[1,1] = cos(rad)
        rotx[1,2] = -sin(rad)
        rotx[2,1] = sin(rad)
        rotx[2,2] = cos(rad)

        return rotx

    def RotY(self,rad):
        '''
            input rad
        '''
        #rad = degTrad(deg)
        roty = np.eye(4)

        roty[0,0] = cos(rad)
        roty[0,2] = sin(rad)
        roty[2,0] = -sin(rad)
        roty[2,2] = cos(rad)

        return roty

    def RotZ(self,rad):
        '''
            input rad
        '''
        #rad = degTrad(deg)
        rotz = np.eye(4)

        rotz[0,0] = cos(rad)
        rotz[0,1] = -sin(rad)
        rotz[1,0] = sin(rad)
        rotz[1,1] = cos(rad)

        return rotz

    def RotXYZ(self,rx=0,ry=0,rz=0):
        '''
            input rad
        '''
        Rotx = self.RotX(rx)
        Roty = self.RotY(ry)
        Rotz = self.RotZ(rz)

        return Rotx @ Roty @ Rotz
  
    def Matrix4(self,x,y,z,rx,ry,rz):
        '''
            rx ry rz input rads
        '''
        trans = self.TransXYZ(x,y,z)
        rotxyz = self.RotXYZ(rx,ry,rz)
        return trans @ rotxyz
  
    def Matrix4_Q(self,q):
        '''
            q = [x,y,z,rx,ry,rz] rx,ry,rz is rads 
        '''
        trans = self.TransXYZ(q[0],q[1],q[2])
        rotxyz = self.RotXYZ(q[3],q[4],q[5])
        return trans @ rotxyz
    
    # rotx(alpha)Transx(a)Transz(d)rotz(θ)
    def RTTR(self,dh):
        '''
            input rad
        '''
        rotx = self.RotX(dh[0])
        transx = self.TransXYZ(dh[1],0,0)
        transz = self.TransXYZ(0,0,dh[2])
        rotz = self.RotZ(dh[3])
        RTTR = np.eye(4)

        RTTR = (((rotz @ transz) @ transx) @ rotx)

        return RTTR

    # rotx(alpha)Transx(a)Transz(d)rotz(θ)
    def RTTR(self,alpha = 0,a = 0 ,d = 0,theta = 0):
        '''
            input rad
        '''

        rotx = self.RotX(alpha)
        transx = self.TransXYZ(a,0,0)
        transz = self.TransXYZ(0,0,d)
        rotz = self.RotZ(theta)
        RTTR = np.eye(4)

        RTTR = (((rotz @ transz) @ transx) @ rotx)

        return RTTR
    
    def Get_Vector(self,crood):
        '''
            input crood
            output 1D 1 * 12 array

            ex:
            4X4 crood
            [Xx,Yx,Zx,Px]
            [Xy,Yy,Zy,Py]
            [Xz,Yz,Zz,Pz]
            [0,0,0,1]

            output = [Xx,Xy,Xz,Yx,Yy,Yz,Zx,Zz,Px,Py,Pz]
        '''
        return crood[:,:3].T.reshape(-1)

    def Get_Matrix(self,A,B,lamda):
        '''
            return [A ~ B] matrix lamda
        '''

        D = np.linalg.inv(A) @ B
        theta = math.acos((D[0,0] + D[1,1] + D[2,2]-1)/2)

        if round(sin(theta * pi),4) != round(0.000,4):
            u = 2 * sin(theta)
            kx = (D[2,1] - D[1,2]) / u
            ky = (D[0,2] - D[2,0]) / u
            kz = (D[1,0] - D[0,1]) / u
        else:
            u = 0.001
            kx = (D[2,1] - D[1,2]) / u
            ky = (D[0,2] - D[2,0]) / u
            kz = (D[1,0] - D[0,1]) / u

        OUT = np.eye(4)

        dx = lamda * D[0,3]
        dy = lamda * D[1,3]
        dz = lamda * D[2,3]

        C = cos(lamda * theta)
        S = sin(lamda * theta)  
        V = 1.0 - C


        OUT[0,0] = kx * kx * V + C
        OUT[0,1] = kx * ky * V - kz * S
        OUT[0,2] = kx * kz * V + ky * S
        OUT[0,3] = dx
        
        OUT[1,0] = kx * ky * V + kz * S 
        OUT[1,1] = ky * ky * V + C
        OUT[1,2] = ky * kz * V - kx * S
        OUT[1,3] = dy

        OUT[2,0] = kx * kz * V - ky * S
        OUT[2,1] = ky * kz * V + kx * S
        OUT[2,2] = kz * kz * V + C
        OUT[2,3] = dz

        #Convert OUT TO PA frame
        OUT = A @ OUT 

        return OUT

    def Get_Matrix_Trajectory(self,A,B,step):
        '''
            return mats [A ~ B] lens step +1
        '''

        D = np.linalg.inv(A) @ B
        theta = math.acos((D[0,0] + D[1,1] + D[2,2]-1)/2)

        if round(sin(theta * pi),4) != round(0.000,4):
            u = 2 * sin(theta)
            kx = (D[2,1] - D[1,2]) / u
            kz = (D[1,0] - D[0,1]) / u
        else:
            u = 0.001
            kx = (D[2,1] - D[1,2]) / u
            ky = (D[0,2] - D[2,0]) / u
            kz = (D[1,0] - D[0,1]) / u

        OUTs= np.zeros((step + 1,4,4))
        rates = np.arange(0,1+1/step,1/step)
        us = np.ones((step+1))
        Cs = cos(rates)
        Ss = sin(rates)
        Vs = 1 - Cs

        dxs = rates * D[0,3]
        dys = rates * D[1,3]
        dzs = rates * D[2,3]

        OUTs[:,0,0] = kx * kx * Vs + Cs
        OUTs[:,0,1] = kx * ky * Vs - kz * Ss
        OUTs[:,0,2] = kx * kz * Vs + ky * Ss
        OUTs[:,0,3] = dxs 

        OUTs[:,1,0] = kx * ky * Vs + kz * Ss
        OUTs[:,1,1] = ky * ky * Vs + Cs
        OUTs[:,1,2] = ky * kz * Vs - kx * Ss
        OUTs[:,1,3] = dys

        OUTs[:,2,0] = kx * kz * Vs - ky * Ss 
        OUTs[:,2,1] = ky * kz * Vs - kx * Ss 
        OUTs[:,2,2] = kz * kz * Vs + Cs 
        OUTs[:,2,3] = dzs

        OUTs[:,3,3] = us
        #convert to A frame 
        OUTs = A @ OUTs 
        return OUTs

    def Fk(self,rad):
        '''
            input rad
        '''
        # T1 = self.RTTR(pi/2,0,0,0)
        T1 = self.RTTR(0,0,0,0)
        T2 = self.RTTR(0,self.L1,0,rad[0])
        T3 = self.RTTR(0,self.L2,0,rad[1])

        P1 = T1 @ T2
        end = P1 @ T3

        Pos = [T1,P1,end]

        return Pos

    def Get_J(self,Pos):
        J = np.zeros((12,2))
        end = Pos[-1]
        for i in range(2):

            zn = Pos[i][:3,2]
            rn = end[:3,3] - Pos[i][:3,3]

            #第i個轉軸對末端點正X方向軸所形成的線速度 vx
            J[0:3,i] = np.cross(zn,end[:3,0])
            #vy
            J[3:6,i] = np.cross(zn,end[:3,1])
            #vz
            J[6:9,i] = np.cross(zn,end[:3,2])
            #v
            J[9:12,i] = np.cross(zn,rn)

        return J
    
    def Jacoiban(self,rads,Pos):
        
        dt = 0.01
        end = Pos[-1]
        n = len(rads)
        J = np.zeros((12,n))
        
        for i in range(n):
            n_rad = copy.copy(rads)
            n_rad[i] += dt
            n_end = self.Fk(n_rad)[-1]
            d_end = n_end - end
            J[:,i] = d_end.T[:,:3].reshape(-1) / dt
        
        return J
             
    def IK(self,src,goal):
        '''
            goal = wTgoal
            Pos[-1] => end => srcTend
            srcTgoal = (src)-1 * goal
            V = srcTgoal - srcTend
        '''

        sTgoal = np.linalg.inv(src) @ goal

        iter = 1000
        goal_rads = np.zeros(2)

        goal_rads[0] = degTrad(1)
        goal_rads[1] = degTrad(1)

        alpha = 0.98

        while True:
            iter -= 1
            Pos = self.Fk(goal_rads)
            V =  sTgoal - Pos[-1]
            V = V.T[:,:3].reshape(-1) 
            error = np.sum(V ** 2)

            if (error <= 0.001) or (iter == 0):
                break

            J = self.Get_J(Pos)
            #V = Jw w = (J)-1 * V
            w = np.linalg.pinv(J) @ V
            goal_rads = goal_rads +  alpha * w  

        print("iter",iter)
        return goal_rads
    
    def IK_(self,rads,src,goal):
        '''
            goal = wTgoal
            Pos[-1] => end => srcTend
            srcTgoal = (src)-1 * goal
            V = srcTgoal - srcTend
        '''

        sTgoal = np.linalg.inv(src) @ goal

        iter = 1000
        goal_rads = rads
        alpha = 0.98

        while True:
            iter -= 1
            Pos = self.Fk(goal_rads)
            V =  sTgoal - Pos[-1]
            V = V.T[:,:3].reshape(-1) 
            error = np.sum(V ** 2)

            if (error <= 0.001) or (iter == 0):
                break

            J = self.Jacoiban(goal_rads,Pos)
            #V = Jw w = (J)-1 * V
            w = np.linalg.pinv(J) @ V
            goal_rads = goal_rads +  alpha * w
            print(goal_rads)  

        print("iter",iter)
        return goal_rads
    
    def Mass(self,q):
        ''' 
            use this method to clculate numpy input rad , return Mass_Matrix
        '''
        
        m1 = self.m1
        m2 = self.m2 
        l1 = self.L1
        l2 = self.L2
        inputq = (q[0],q[1])
        input_mass = (m1,m2)
        M = self.__M(([input_mass],l1,l2),[inputq])
        
        return M

    def Coriolis(self,q,qd):
        ''' 
            use this method to clculate numpy input rad , return Coriolis_Matrix
        '''
        m1 = self.m1
        m2 = self.m2 
        l1 = self.L1
        l2 = self.L2
        inputq = (q[0],q[1])
        inputqd = (qd[0],qd[1])
        input_mass = (m1,m2)
        
        V = self.__V(([input_mass],l1,l2),[inputq],[inputqd])
        
        return V
    
    def Gravity(self,q):
        
        m1 = self.m1
        m2 = self.m2 
        l1 = self.L1
        l2 = self.L2
        inputq = (q[0],q[1])
        g = -9.8
        input_mass = (m1,m2)
        
        G = self.__G(([input_mass],l1,l2,g),[inputq])
        return G
 
    def Euler_Integral(self,q0,qd0,qdd,dt):
        '''
            x(t+1) = x0 + v * dt
        '''
        qd = qd0 + qdd * dt
        q = q0 + qd * dt 
        
        return q,qd 
    
    def Runge_Kutta4_Integral(self,q0,qd0,torque,h,Ftip = [0,0,0,0,0,0]):
        '''
            High precision Integral Method
        '''
          
        k1v = self.ForwardDynamics(torque,q0,qd0,Ftip)
        k1v = np.clip(k1v,-28/15*pi,28/15*pi)
        k2v = self.ForwardDynamics(torque,q0,qd0 + 0.5 * h * k1v,Ftip)
        k2v = np.clip(k2v,-28/15*pi,28/15*pi)
        
        k3v = self.ForwardDynamics(torque,q0,qd0 + 0.5 * h * k2v,Ftip)
        k3v = np.clip(k3v,-28/15*pi,28/15*pi)
        
        k4v = self.ForwardDynamics(torque,q0,qd0 + h * k3v, Ftip )
        k4v = np.clip(k4v,-28/15*pi,28/15*pi)
        
        qdd = k1v
        qd = qd0 + h/6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        qd = np.clip(qd,-28/15*pi,28/15*pi) * 0.99
        q  = q0 + qd * h

        return q,qd,qdd

    def InverseDynamics(self,q,qd,qdd,Ftip = [0,0,0,0,0,0]):
        '''
            use Lagrange equation Create 
            torque = M * qdd + C * qd + G + (JT*fend)

            Mass subs q
            Coriolis subs q , qd
            G subs g,q 

            q,qd,qdd is input rad,rad/s,rad/s^2
        '''
        M,V,G,JT = self.__CalCulate_Dynamics_Numpy(q,qd)

        torque = np.dot(M,qdd) + V[:,0] + G[:,0] + JT @ Ftip

        return torque

    def ForwardDynamics(self,torque,q,qd,Ftip = [0,0,0,0,0,0]):
        '''
            tor = M @ dqq + V + G + JT @ Ftip
            M @ dqq = tor - V - G - JT @ Ftip
            dqq = pinv(M) @ (tor - V - G - JT*Ftip)
            q,qd is input rad,rad/s
        '''

        n = len(q)
        torque = np.array(torque)
        M,V,G,JT = self.__CalCulate_Dynamics_Numpy(q,qd)
        M_ = np.linalg.pinv(M)
        Mdqq = torque - V[:,0] - G[:,0] - JT @ Ftip
        qdd = np.dot(M_,Mdqq)
        
        return qdd
  
    def __CalCulate_Dynamics_Numpy(self,q,qd):
        ''' 
            use this method to clculate numpy input rad
        '''
        
        m1 = self.m1
        m2 = self.m2 
        l1 = self.L1
        l2 = self.L2
        inputq = (q[0],q[1])
        inputqd = (qd[0],qd[1])
        input_mass = (m1,m2)
        g = -9.8
        
        M = self.__M(([input_mass],l1,l2),[inputq])
        V = self.__V(([input_mass],l1,l2),[inputq],[inputqd])
        G = self.__G(([input_mass],l1,l2,g),[inputq])
        J = self.__J(([input_mass],l1,l2),[inputq])
        J_T = np.transpose(J)

        return M,V,G,J_T

    def __Create_Dynamics_symbol(self):
        '''
            this area can change for each robot
        '''
        n = 2
        tool = Robot_Sympy()
        l1 = sympy.Symbol('l1')
        l2 = sympy.Symbol('l2')
        g = sympy.Symbol('g')
        mass = sympy.symbols(f"m1:{n+1}")
        q = sympy.symbols(f'q1:{n+1}')      #input angle
        qd = sympy.symbols(f'qd1:{n+1}')    #input angle vel
        qdd = sympy.symbols(f'qdd1:{n+1}')  #input angle acc
        I = [self.I1,self.I2]
        Pc1 = sympy.eye(4) 
        Pc1[0,3] = self.L1 /2
        Pc2 = sympy.eye(4)
        Pc2[0,3] = self.L2 /2
        Pc = [Pc1,Pc2]
        rotate_axis = ['z','z']

        # T1 = tool.RTTR_Matrix(pi/2,0,0,0)
        T1 = tool.RTTR_Matrix(0,0,0,0)
        T2 = tool.RTTR_Matrix(0,l1,0,q[0])
        T3 = tool.RTTR_Matrix(0,l2,0,q[1])
        Ti = []
        Ti.append(T1 @ T2 )
        Ti.append(T1 @ T2 @ T3)
        
        #genarator sympy term
        M,V,C,G,Jv,Jw,J,KE,Ki,PE = tool.Lagarange(n,q,qd,Ti,rotate_axis,mass,Pc,'z',I)

        self.__Mass_Matrix = M
        self.__Coriolis_Matrix = C
        self.__Coriolis_V_Matrix = V
        self.__Gravity = G
        self.__Kinetic_energy = KE
        self.__Ki = Ki
        self.__Potential_energy = PE
        self.__Jacobian_linear_vel = Jv
        self.__Jacobian_angluar_vel = Jw
        self.__Jacobian = J

        #use this for clc
        self.__M = sympy.lambdify([([(mass)],l1,l2),[(q)]],self.__Mass_Matrix,"numpy")
        self.__V = sympy.lambdify([([(mass)],l1,l2),[(q)],[(qd)]],self.__Coriolis_V_Matrix,"numpy")
        self.__G = sympy.lambdify([([(mass)],l1,l2,g),[(q)]],self.__Gravity,"numpy")
        self.__J = sympy.lambdify([([(mass)],l1,l2),[(q)]],self.__Jacobian,"numpy")
    
    def Display_Dynamics_eq(self):
        '''
            display Dynamics equation 
            Mass , Coriolis Matrix C , Coriolis Matrix V is np.dot(C,qd) , Gravity Matrix G
        '''
        from IPython.display import display
        display(self.__Mass_Matrix)
        display(self.__Coriolis_Matrix)
        display(self.__Coriolis_V_Matrix)
        display(self.__Gravity)

        display(self.__M)
        display(self.__V)
        display(self.__G)
    


#%%
if __name__ == "__main__":
    test = RR(1,1)
    test.Init_Pos()
    #test.Display_Dynamics_eq()
    Pos = test.Fk([pi/3,pi/3])
    print(Pos[-1])
    # n = 2

    # torque = test.InverseDynamics([0,0],[0,0],[0,0])
    # torque_initial = np.array([0,0]).reshape(n,1)
    # q_initial = np.array([0,0]).reshape(n,1)
    # qd_initial = np.array([0,0]).reshape(n,1)
    # q_list = []
    # qd_list = []
    # qdd_list = []
    # plotq1 = []
    # plotqd1 = []
    # plotq2 = []
    # plotqd2= []
    # q_list.append(q_initial)
    # qd_list.append(qd_initial)
    # qdd_initial = test.ForwardDynamics(torque_initial,q_initial,qd_initial)
    # qdd_initial = radTdeg(qdd_initial) % 360
    # qdd_list.append(qdd_initial)
    # for i in range(5000):
    #     qd = qdd_list[-1] * test.dt + qd_list[-1]
    #     q = qd * test.dt + q_list[-1]
    #     qd = radTdeg(qd) % 360
    #     q = radTdeg(q) % 360
    #     q_list.append(q )
    #     qd_list.append(qd)

    #     torque = test.InverseDynamics(q_list[-1],qd_list[-1],qdd_list[-1])
    #     qdd = test.ForwardDynamics(torque_initial,q_list[-1],qd_list[-1])
    #     #qdd = test.ForwardDynamics(torque,q_list[-1],qd_list[-1])
    #     qdd = radTdeg(qdd) % 360
    #     qdd_list.append(qdd)

    #     Pos = test.Fk(q_list[-1])
    #     test.Fixed_Src(Pos)
    #     plt.cla()
    #     plt.plot(test.p1,test.end)
    #     # plt.plot()
    #     plt.pause(0.01)
        

 
    #     # plotq1.append(q[0,0])
    #     # plotqd1.append(qd[0,0])
    #     # plotq2.append(q[1,0])
    #     # plotqd2.append(qd[1,0])
    # plt.ioff()
    # plt.show()
    
    # plt.plot(plotq)
    # plt.plot(plotqd)
    # fig1= plt.figure()
    # fig2= plt.figure()

    # ax1 = fig1.add_subplot(1,1,1)
    # ax2 = fig2.add_subplot(1,1,1)
    
    

    # ax1.plot(plotq1,plotqd1)
    # ax2.plot(plotq2,plotqd2)
    
    # # plt.plot(qdd)
    # plt.show()

    #print(torque)

# %%
