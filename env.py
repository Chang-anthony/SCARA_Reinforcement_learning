#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re
import sys 
import os 
import csv 
import os.path
from OpenGL.GL import * 
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5 import QtCore,QtGui,QtWidgets,QtOpenGL
from PyQt5.QtWidgets import QDialog,QApplication,QMessageBox,QMainWindow,QAction
import matplotlib.pyplot as plt
from motor import Motor

import debugpy
import cv2
import numpy as np
import time
import queue
import copy
from RR import RR
from PID import PID
from Matrix4_4 import Matrix4


#自定義numpy 函式 
# np.set_printoptions(precision=3,suppress=True)
cos = np.cos
sin = np.sin
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg


class RREnv(object):
    viewer = None #初始化viewer,為沒有
    dt = 0.04 #refresh time
    state_dimension = 9 #有幾個狀態可以觀察
    action_dimesion = 2 #有幾個動作可選
    done = False
    move_counter = 0
    q_bound = [-pi,pi] #rad
    # qd_bound = [-28/15*pi,28/15*pi] #mx106 max_rpm => 56 max_rps => 56/60 rps max_rad/s => 56/60 * 2pi 
    qd_bound = [-28/15*pi,28/15*pi]
    goal = None
    

    def __init__(self,RR:RR):
        self.RR = RR
        self.queuy_gl = queue.Queue()
        self.rads = np.zeros(2)
        self.__drads = np.zeros(2)
        self.__ddrads = np.zeros(2)
        self.__torque = np.zeros(2)
        self.__Q = np.diag([5,5,0.001,0.001])
        self.__R = np.diag([1,1])
        
        #init motor
        self.ID1 = Motor()
        self.ID2 = Motor()
        self.ID1.reset(855,0,0,self.rads[0],self.__drads[0])
        self.ID2.reset(855,0,0,self.rads[1],self.__drads[1])
          
    def step(self,action):
        '''
            this area need to fix
        '''
        q = self.rads
        qd = self.__drads
        qdd_d = self.__ddrads
        
        input_q = np.copy(action) 
        input_qd = np.copy(action) / self.dt
        input_qd = np.clip(input_qd,self.qd_bound[0],self.qd_bound[1])
        
        now_torque = self.RR.InverseDynamics(q,qd,qdd_d)
        

        torque1 = self.ID1.update(input_q[0],input_qd[0],q[0],qd[0],now_torque[0])
        torque2 = self.ID2.update(input_q[1],input_qd[1],q[1],qd[1],now_torque[1])

        input_torque = np.array([torque1,torque2])
        new_q,new_qd,qdd_u = self.RR.Runge_Kutta4_Integral(q,qd,input_torque,self.dt)
        
         
        # qdd_u = self.RR.ForwardDynamics(input_torque,q,qd)
        # new_q,new_qd = self.RR.Euler_Integral(q,qd,qdd_u,self.dt)
        
        self.rads = self.__clip(new_q)
        self.__drads = new_qd 
        self.__ddrads = qdd_u 
        

        end = self.__get_End_point()
        reward = self.reward(end)
        reward -= np.sum(abs(qdd_u)) * 0.001
        # reward -=  0.002 * np.sum(abs(self.__drads)) 
        # reward = self.__clculate_tor_reward(end,input_torque)

        
        # t_arms = np.ravel(end[:3,3] - self.RR.src[:3,3])
        t_arms = end[:3,3]
        state = np.hstack((self.rads,self.__drads,self.__ddrads,t_arms))
        # state = self.state_normlize(state)
        
        return state,reward,self.done
    
    def state_normlize(self,state):
        state_mean = np.mean(state)
        state_std = np.std(state)
        state = (state - state_mean) / (state_std + 1e-5)
        
        return state
    
    def __clip(self,rads):
        rads = rads % (2 * pi)
        return rads
          
    #初始化
    def reset(self):
        '''
            return rads
        '''
        self.done = False
        self.move_counter = 0
        self.rads = np.random.rand(2) * pi * 0.5 
        self.rads = self.__clip(self.rads) 
        self.__drads = np.zeros(2)
        self.__ddrads = np.zeros(2)
        self.__past_distance = None
        self.circle_count = 0
        
        self.ID1.reset(855,0,0,self.rads[0],self.__drads[0])
        self.ID2.reset(855,0,0,self.rads[1],self.__drads[1])
        
        #initial
        Pos = self.RR.Fk(self.rads)
        self.RR.Fixed_Src(Pos)
        
        errq = (np.sum(self.goal_q - np.clip(self.rads,-100,100)) ** 2) ** 0.5
        if errq <= 0.03:
            self.reset()
        
        end = self.__get_End_point()
        # t_arms = np.ravel(end[:3,3] - self.RR.src[:3,3])
        t_arms = end[:3,3]
        state = np.hstack((self.rads,self.__drads,self.__ddrads,t_arms))
        # state = self.state_normlize(state)
        
        return state
    
    def render(self):
        if self.viewer is None: #如果調用了 render,而且沒有viewer就生成一個
            self.viewer = Viewer3D(self.queuy_gl,self.RR)
        self.viewer.render()
        
    def simple_action(self):
        action = np.random.rand(2) * 2 - 0.5
        return action
    
    def set_goal(self,goal_point,goal_qd = None):
        '''
            input Q [x,y,z,rx,ry,rz]
            is_trajectory  
        '''
        self.goal = self.RR.Matrix4_Q(goal_point)
        self.goal_q = self.RR.IK_(self.rads,self.RR.src,self.goal)
        self.goal_qd = np.zeros(2)
        
    def set_goal_trajectory(self,goal_point,step):
        '''
            need to fix
        '''
        goal = self.RR.Matrix4_Q(goal_point)
        self.RR.Get_Matrix_Trajectory(self.RR.src,goal,step)
    
    def __get_End_point(self):
        Pos = self.RR.Fk(self.rads)
        self.RR.Fixed_Src(Pos)
        
        self.__push_queuy_gl()
        return self.RR.end
       
    def __push_queuy_gl(self):
        data = [self.RR.src,
                self.RR.p1,
                self.RR.end,
                self.goal
            ]

        self.queuy_gl.put(data)
           
    def reward(self,end):
        distance = np.sum((self.goal - end).T[3,:3].reshape(-1)**2)**0.5
        
        distance = (self.goal - end).T[3,:3]
        errq = (np.sum(self.goal_q - np.clip(self.rads,-100,100)) ** 2) ** 0.5
        r = 0
        distance = np.linalg.norm(distance)
        
        if self.__past_distance is None:
            self.__past_distance = distance
            dx = 0
        else:
            dx = distance - self.__past_distance
            self.__past_distance = distance 
        
        if distance >= 2.5:
            # r = -1.
            self.circle_count += 1
        else:
            self.circle_count = 0
            
        if distance <= 0.05 and self.circle_count < 20:
            r = 10.
            self.done = True
        if self.circle_count >= 20:
            print("bad")
            r = -10.
            self.done = True
            
        # distance = 1 /( 1 + distance)
            
        return -distance - 0.5 * dx + r
    
    
class OpenGL_widget(QThread):
    def __init__(self,robotpoints,RR:RR):
        super().__init__()
        self.query_gl = robotpoints
        self.shadow_robot = RR     
        self.dx = 1
        self.x1 = 0
        self.ts = []

        self.src = np.eye(4)
        self.p1 = np.eye(4)
        self.end = np.eye(4)

        self.stop = True
        self.window = None

        '''
            cammat 攝影機其次座標矩陣
            campos 攝影機在世界座標的x,y,z位置
            camlook 攝影機看向的世界座標點
        '''
        self.cammat = np.eye(4)
        self.cammat[:3,3] = np.array([50,50,50])
        self.camlook = np.array([0,0,10])
        Zaxis = np.array([0,0,1])


        P = self.cammat[:3,3]
        A = self.camlook
        forward = A-P
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward,Zaxis)
        right = right / np.linalg.norm(right)
        Top = np.cross(right,forward)
        Top  = Top / np.linalg.norm(Top)

        '''
            camera Xaxis right
            camera Yaxis -Top
            camera Zaxis forward
        '''

        self.cammat[:3,0] = right
        self.cammat[:3,1] = -Top
        self.cammat[:3,2] = forward

    def close(self):
        if self.window:
            print("exit OpenGL")

    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
        glutInitWindowPosition(0,0)
        glutInitWindowSize(640,360)

        self.window = glutCreateWindow("RR_sim")

        # glutMouseWheelFunc(self.mouseWheel)
        glutKeyboardFunc(self.keyboard_envent)
        glutMouseFunc(self.mouse_event)
        glutDisplayFunc(self.paintGL)
        glutDialsFunc(self.paintGL)
        glutIdleFunc(self.paintGL)

        glClearColor(0,0,0,1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glutMainLoop()
        glutMainLoopEvent()

    def TransXYZ(self,dx = 0, dy = 0,dz =0):
        Trans = np.eye(4)

        Trans[0,3] = dx
        Trans[1,3] = dy
        Trans[2,3] = dz

        return Trans

    def RotX(self,rad):
        '''
            input rad
        '''
        #rad = degTrad(deg)
        RotX = np.eye(4)

        RotX[1,1] = cos(rad)
        RotX[1,2] = -sin(rad)
        RotX[2,1] = sin(rad)
        RotX[2,2] = cos(rad)

        return RotX

    #右手座標係
    def RotY(self,rad):
        '''
            input rad
        '''
        #rad = degTrad(deg)
        RotY= np.eye(4)

        RotY[0,0] = cos(rad)
        RotY[0,2] = sin(rad)
        RotY[2,0] = -sin(rad)
        RotY[2,2] = cos(rad)
        
        return RotY

    def RotZ(self,rad):
        '''
            input rad
        '''
        #rad = degTrad(deg)
        RotZ = np.eye(4)
        
        RotZ[0,0] = cos(rad)
        RotZ[0,1] = -sin(rad)
        RotZ[1,0] = cos(rad)
        RotZ[1,1] = sin(rad)

        return RotZ

    def RotXYZ(self,rx = 0 ,ry = 0,rz = 0):
        '''
            input rad
        '''
        RotX = self.RotX(rx)
        RotY = self.RotY(ry) 
        RotZ = self.RotZ(rz)

        return RotX @ RotY @ RotZ

    def paintGL(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45,640/360,0.01,5000)

        scale = 10
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


        self.campos = self.cammat[:3,3]
        self.camlook = np.array(self.src[:3,3])

        P = self.cammat[:3,3]
        A = self.camlook
        forward = (A-P)
        Zaxis = np.array([0,0,1])

        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward,Zaxis)
        right = right / np.linalg.norm(right)
        Top = np.cross(right,forward)
        Top = Top / np.linalg.norm(Top)

        self.cammat[:3,0] = right
        self.cammat[:3,1] = -Top
        self.cammat[:3,2] = forward

        gluLookAt(P[0],P[1],P[2],A[0],A[1],A[2],Top[0],Top[1],Top[2])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        '''
        點 : GL_POINTS
        線 : GL_LINES
        連續線 : GL_LINE_STRIP
        封閉線 : GL_LINE_LOOP
        獨立三角形 : GL_TRIANGLES
        連續三角形 : GL_TRIANGLE_STRIP
        三角形扇面 : GL_TRIANGLE_FAN
        4 GL_QUADS
        '''

        glPushMatrix()
        glLineWidth(10.0)

        glBegin(GL_LINES)
        glColor3f(1.0,0.0,0.0) #RGB
        #world Xaxis
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(10.0,0.0,0.0)
        #world Yaxis
        glColor3f(0.0,1.0,0.0)
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(0.0,10.0,0.0)
        #world Zaxis
        glColor3f(0.0,0.0,1.0)
        glVertex3f(0.0,0.0,0.0)
        glVertex3f(0.0,0.0,10.0)
        glEnd()

        glPopMatrix()

        if not self.query_gl.empty():
            data = self.query_gl.get()
            self.src = data[0]
            self.p1 = data[1]
            self.end = data[2]
            self.goal = data[3]
        


        self.draw_ground()
        self.draw_robot()

        if self.window:
            glutSwapBuffers()
        self.ts.append(time.time())

        if len(self.ts) >= 2:
            dt = self.ts[-1] - self.ts[-2]

    def mouseWheel(self,button,dir,x,y):
        if (dir > 0):
            #Zoom in 
            self.cammat = self.cammat @ self.TransXYZ(dz = -0.5)
        else:
            self.cammat = self.cammat @ self.TransXYZ(dz = 0.5)
    
    def keyboard_envent(self,c,x,y):
        print("enter" ,ord(c) ,x ,y)

        if c == 27:
            print("exit")
        if ord(c) == ord('a'):
            self.cammat = self.cammat @ self.TransXYZ(dx = -2.5)
        if ord(c) == ord('d'):
            self.cammat = self.cammat @ self.TransXYZ(dx = 2.5) 
        if ord(c) == ord('w'):
            self.cammat = self.TransXYZ(dz = 2.5) @ self.cammat
        if ord(c) == ord('s'):
            self.cammat = self.TransXYZ(dz = -2.5) @ self.cammat

    def mouse_event(self,button,state,x,y):
        if button == GLUT_LEFT_BUTTON:
            if (state == GLUT_DOWN):
                print("LB_DOWN x:" ,x ,"y",y)
        elif button == GLUT_RIGHT_BUTTON:
            if (state == GLUT_DOWN):
                print("RB_DOWN x:" ,x ,"y",y)
        elif button == GLUT_MIDDLE_BUTTON:
            if (state == GLUT_DOWN):
                print("MB_DOWN x:" ,x ,"y",y)
        elif button == 3:
            print("MOUSE_WHEEL_UP")
            self.cammat = self.cammat @ self.TransXYZ(dz = -0.5)
        elif button == 4:
            print("MOUSE_WHEEL_DOWN")
            self.cammat = self.cammat @ self.TransXYZ(dz = 0.5)

    def draw_tag(self,wTq):
        wTq[:3,3] = wTq[:3,3] * 0.1
        q = self.draw_crood(wTq,5)

    def draw_robot(self,line_width = 5.0):
        
        src = self.getpos(self.src)
        p1 =  self.getpos(self.p1)
        end = self.getpos(self.end)
        # src = self.draw_crood(self.src)
        # p1 = self.draw_crood(self.p1) 
        # end = self.draw_crood(self.end)
        
        scale = 5
        glLineWidth(line_width)
        glScaled(scale,scale,scale)
        goal = self.draw_crood(self.goal,1)
        #glScaled(0.001,0.001,0.001)
        glBegin(GL_LINES)
        glColor3f(1.0,0.0,0.0)
        glVertex3f(0,0,0)
        glVertex3f(src[0],src[1],src[2])
        
        glColor3f(0.0,1.0,0.0)
        glVertex3f(src[0],src[1],src[2])
        glVertex3f(p1[0],p1[1],p1[2])

        glColor3f(0.0,0.0,1.0)
        glVertex3f(p1[0],p1[1],p1[2])
        glVertex3f(end[0],end[1],end[2])
        
        glEnd()

    def draw_crood(self,crood,scale = 3):
        glBegin(GL_LINES)
        pos = crood[:3,3]
        xpos = pos + crood[:3,0] * scale
        ypos = pos + crood[:3,1] * scale
        zpos = pos + crood[:3,2] * scale

        glColor3f(1.0,0.0,0.0) #RBG
        glVertex3f(pos[0],pos[1],pos[2])
        glVertex3f(xpos[0],xpos[1],xpos[2])

        glColor3f(0.0,1.0,0.0)
        glVertex3f(pos[0],pos[1],pos[2])
        glVertex3f(ypos[0],ypos[1],ypos[2])

        glColor3f(0.0,0.0,1.0)
        glVertex3f(pos[0],pos[1],pos[2])
        glVertex3f(zpos[0],zpos[1],zpos[2])
        glEnd()

        return pos
    
    def getpos(self,coord):
        pos = coord[:3,3]

        return pos
    
    def draw_ground(self):
        glColor3f(1.0,1.0,1.0)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(21):
            data = -100 + i * 10
            glVertex3f(data,100,0)
            glVertex3f(data,-100,0)

            glVertex3f(100,data,0)
            glVertex3f(-100,data,0)
        glEnd() 
    
class Viewer3D(QDialog):
    def __init__(self,queuy_gl,robot:RR):
        super().__init__()
        self.system_time = 0.04 #40ms

        self.queuy_gl = queuy_gl
        self.queuy_gait = queue.Queue()
        self.robot = robot

        self.view3D = OpenGL_widget(self.queuy_gl,self.robot)
        self.view3D.start()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)# period, in milliseconds

        self.gamethread = None

    def render(self):
        self.show()
        time.sleep(0.05)
        
        
if __name__=="__main__":
    
    app = QApplication(sys.argv)
    robot = RR(1,1,1,1,np.eye(3),np.eye(3))
    env = RREnv(robot)
    
    goal_qd = np.zeros(env.state_dimension)
    goal = [1,0,4,0,0,0]
    env.set_goal(goal,goal_qd) 
    
    
    epoch = 0
    s = env.reset()
    
    while epoch < 1000:
        env.render()
        a = env.simple_action()
        env.step(a)
        epoch += 1
 
    app.exec()
    print("exit_form")