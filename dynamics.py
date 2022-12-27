#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
class Dynamics(object):
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def Mass(q,T_list,mass_list,Pc_list,Inertia_list,rotate_axis):
        '''
        組合慣量法:
            tor = M(ddq)ddq + C(q,dq) + G + ftip
            q = [0,0,0,0,0....n] 有 link n 個
            dq = [0,0,0,0,0....n] 有 link n 個
            跑 link n 次迴圈 每次 ddq(i) 都為 1
            設 i = 0 , link n = 6
            則 ddq = [1 0 0 0 0 0]
            tor = M()*ddq 代入上面 M(:,i) = tor 
        '''
        
        link_n = len(q)
        M = np.zeros((link_n,link_n))
        
        for i in range(link_n):
            ddq = [0] * link_n
            ddq[i] = 1
            
            M[:,i] = Dynamics.newton_euler(q,[0]*link_n,ddq,T_list,mass_list,Pc_list,Inertia_list,rotate_axis)
            
        return M

    def Coriolis(q,dq,T_list,mass_list,Pc_list,Inertia_list,rotate_axis):
        '''
            tor = M(ddq)ddq + C(q,dq) + G(q) + ftip
            使 ddq = [0] * link_n ,G = 0, Ftip = 0
            tor = C(q,dq)
        '''
        
        link_n = len(q)
        C = Dynamics.newton_euler(q,dq,[0]*link_n,T_list,mass_list,Pc_list,Inertia_list,rotate_axis)
            
        return C

    def Gravity(q,T_list,mass_list,Pc_list,Inertia_list,rotate_axis,G):
        '''
            tor = M(ddq)ddq + C(q,dq) + G + ftip
            使 ddq = [0] * link_n , dq = [0] * link_n , G = 1 =>[0 0 -9.8] ftip = 0
            tor = G(q)
        '''
        link_n = len(q)
        G = Dynamics.newton_euler(q,[0]*link_n,[0]*link_n,T_list,mass_list,Pc_list,Inertia_list,rotate_axis,G)
        
        return G

    def Endeffect_force(q,T_list,Ftip,mass_list,Pc_list,Inertia_list,rotate_axis):
        '''
            tor = M(ddq)ddq + C(q,dq) + G + ftip
            使 ddq = [0] * link_n , dq = [0] * link_n , G = 0 ,Ftip = Ftip
            tor = G(q)
        '''

    def Forward_dynamics(q,dq,tau_list,T_list,mass_list,Pc_list,Inertia_list,rotate_axis,G):
        '''
            tor = M(ddq)ddq + C(q,dq) + G(q) + ftip
            M(ddq)ddq = tau - C(q,dq) - G(q) - ftip
            ddq = inv(M(ddq))*(tau - C(q,dq) - G(q) - ftip)
        '''
        Mddq = tau_list - Dynamics.Coriolis(q,dq,T_list,mass_list,Pc_list,Inertia_list,rotate_axis) - Dynamics.Gravity(q,T_list,mass_list,Pc_list,Inertia_list,G)
        M = Dynamics.Mass(q,T_list,mass_list,Pc_list,Inertia_list,rotate_axis)
        ddq = np.linalg.pinv(M) @ Mddq

        return ddq
    
    def newton_euler(q,dq,ddq,T_list,mass_list,Pc_list,Inertia_list,rotate_axis,G=None):
        '''
            if G == 'x':
                v00d = np.array([-9.8,0,0]) 
            if G == 'y':
                v00d = np.array([0,-9.8,0])
            if G == 'z':
                v00d = np.array([0,0,-9.8])
            elif G == None:
                v00d = np.array([0,0,0])
            
            T_list is forward kinmatics Transformatrix
            if Joint is 6
            have [T01,T12,T23,T34,T45,T56]
            
            mass_list = [m1,m2,m3,m4 ...... mn]
            Inertia_list = [I1,I2,I3,....In] 
        '''
        #有幾個旋轉軸
        link_n = len(q)
        z = []
        #這邊都是假設為RTTR 建模，如果不是RTTR要根據所繞的旋轉軸去設
        for i in range(link_n):
            if rotate_axis[i] == 'x':
                rot = np.array([1,0,0])
            if rotate_axis[i] == 'y':
                rot = np.array([0,1,0])
            if rotate_axis[i] == 'z':
                rot = np.array([0,0,1])
            if rotate_axis[i] == '-x':
                rot = -np.array([1,0,0])
            if rotate_axis[i] == '-y':
                rot = -np.array([0,1,0])
            if rotate_axis[i] == '-z':
                rot = -np.array([0,0,1])
            if rotate_axis[i] == 'fixed':
                rot = np.array([0,0,0])
            z.append(rot)        
        
        
        #base_link initial velocity
        w00 = np.array([0,0,0])
        w00d = np.array([0,0,0])
        
        #external force for end-effect對於末端點所施加的外力
        fnn = np.array([0,0,0])
        n77 = np.array([0,0,0])
        
        if G == 'x':
            v00d = np.array([-9.8,0,0]) 
        if G == 'y':
            v00d = np.array([0,-9.8,0])
        if G == 'z':
            v00d = np.array([0,0,-9.8])
        elif G == None:
            v00d = np.array([0,0,0])
        
        Pos_list = []
        for i in range(link_n):
            Pos_list.append(T_list[i][:3,3]) 
        #ecah joint pos
        Pos_list.append(np.array([0,0,0]))
        
        #check
        for i in range(link_n+1):
            print(Pos_list[i])
        
        if Pc_list is None:
            Pc_list = []
            for i in range(link_n):
                Pc = T_list[i][:,:] @ Pc_list[i][:,:]
                Pc_list.append(Pc[:3,3])
        else:
            Pc = []
            for i in range(link_n):
                Pc.append(Pc_list[i][:3,3])
            Pc_list = Pc
        
        for i in range(link_n):
            print(Pc_list[i])

        #inverse Rotate Matrix => R-1 = R.T
        #旋轉矩陣與旋轉矩陣逆(等於轉置)
        R_list = []
        RT_list = []
        for i in range(link_n):
            R = T_list[i][:3,:3]
            R_list.append(R)
            RT_list.append(R.T)
        R_list.append(np.eye(3))
        RT_list.append(np.eye(3).T)
        
        w = []
        wd = []
        vd = []
        vcd = []
        F = []
        N = []
        
        #inital force list and torque list
        #初始化列表長度
        f = [None] * link_n
        n = [None] * link_n
        tor = [None] * link_n
        
        
        #forward recursion
        for i in range(link_n):
            if i == 0:
                w11 = RT_list[i] @ w00 + dq[i] * z[i] 
                w11d = RT_list[i] @ w00d + np.cross(RT_list[i] @ w00, z[i] * dq[i]) + ddq[i] * z[i]
                v11d = RT_list[i] @ (np.cross(w00d,Pos_list[i])+ np.cross(w00,np.cross(w00,Pos_list[i])) + v00d)
                vc11d = np.cross(w11d,Pc_list[i]) + np.cross(w11,np.cross(w11,Pc_list[i])) + v11d
                F11 = mass_list[i] * vc11d
                N11 = Inertia_list[i] @ w11d + np.cross(w11,Inertia_list[i] @ w11)
                w.append(w11)
                wd.append(w11d)
                vd.append(v11d)
                vcd.append(vc11d)
                F.append(F11)
                N.append(N11)
            else:
                wi = RT_list[i] @ w[i-1] + dq[i] * z[i]
                wid = RT_list[i] @ wd[i-1] + np.cross(RT_list[i] @ w[i-1], z[i] * dq[i]) + ddq[i] * z[i]   
                vid = RT_list[i] @ (np.cross(wd[i-1],Pos_list[i])+ np.cross(w[i-1],np.cross(w[i-1],Pos_list[i])) + vd[i-1])
                w.append(wi)
                wd.append(wid)
                vd.append(vid)
                vcid = np.cross(wd[i],Pc_list[i]) + np.cross(w[i],np.cross(w[i],Pc_list[i])) + vd[i]
                vcd.append(vcid)
                Fi = mass_list[i] * vcd[i]
                Ni = Inertia_list[i] @ wd[i] + np.cross(w[i],Inertia_list[i] @ w[i])
                F.append(Fi)
                N.append(Ni)
                
        
        #Backward recursion n -> 1 range(開始,結束,每次減少數量)
        for i in range(link_n,0,-1):
            if i == link_n:
                fi = R_list[i] @ fnn + F[i-1]
                ni = R_list[i] @ n77 + N[i-1] + np.cross(Pc_list[i-1],F[i-1]) + np.cross(Pos_list[i],R_list[i] @ fnn)
                f[i-1] = fi
                n[i-1] = ni
                tor[i-1] = ni @ z[i-1]
            else:
                fi = R_list[i] @ f[i] + F[i-1]
                ni = R_list[i] @ n[i] + N[i-1] + np.cross(Pc_list[i-1],F[i-1]) + np.cross(Pos_list[i],R_list[i] @ f[i])
                f[i-1] = fi
                n[i-1] = ni
                tor[i-1] = ni @ z[i-1]
        
        return tor

    def Inertia(ixx,iyy,izz,ixy,ixz,iyz):
        I = np.array([[ixx,ixy,ixz],
                        [ixy,iyy,izz],
                        [ixz,iyz,izz]])
        
        return I


    
    
