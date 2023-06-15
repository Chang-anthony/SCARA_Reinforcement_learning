#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : anthony
"""

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

import debugpy
import cv2
import numpy as np
import time
import queue
import copy
from RR import RR
from PID import PID



#自定義numpy 函式 
# np.set_printoptions(precision=3,suppress=True)
cos = np.cos
sin = np.sin
pi = np.pi
degTrad = np.deg2rad
radTdeg = np.rad2deg

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

        glClearColor(0.5,0.5,0.5,1)
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
        glLineWidth(3.0)
        glBegin(GL_LINES)
        for i in range(21):
            data = -100 + i * 10
            glVertex3f(data,100,0)
            glVertex3f(data,-100,0)

            glVertex3f(100,data,0)
            glVertex3f(-100,data,0)
        glEnd() 

class RobotAPP(QDialog):
    def __init__(self):
        super().__init__()
        self.system_time = 0.04 #40ms

        self.queuy_gl = queue.Queue()
        self.queuy_gait = queue.Queue()
        self.robot = RR(1,1,1,1)

        self.view3D = OpenGL_widget(self.queuy_gl,self.robot)

        self.Workready()
        self.view3D.start()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)# period, in milliseconds

        self.gamethread = None

    def Init(self):
        degs = self.robot.Init_Pos()
        self.Draw()

    def Workready(self):
        self.robot.ResetPos()
        self.Draw()
        
    def PID(self,goalq,goal_qd,q_in,qd_in,torque_in):
        '''
            goal => 
        '''
        print("goal",radTdeg(goalq))
        qdd_in = self.robot.ForwardDynamics(torque_in,q_in,qd_in)
        q_list = [q_in]
        qd_list = [qd_in]
        qdd_list = [qdd_in]
        errs = []
        
        Km = 1
        torque_d = self.robot.InverseDynamics(goalq,goal_qd,[0,0]) * Km

        q1_pid = PID(KP = 1000.0,KI=5.0,KD = 1.0,action_bound=[-50,50])
        q2_pid = PID(KP = 100.0,KI=1.0,KD = 1.0,action_bound=[-20,20])
        
        KU1 = 10
        KU2 = 10
        TU1 = 12.2 - 10.0
        TU2 = 10.16 - 8.18
        
        # q1_pid.PID_ZN(KU1,TU1)
        # q2_pid.PID_ZN(KU2,TU2)
        
        torque1 = 0
        torque2 = 0
 
        plotq1 = []
        plotq2 = []
        ts = []
        epoc = 0
        max_epoc = 2000
        
        qdd_d = np.array([0,0])
        
        KP_test = 50
        KD_test = 18
        KI_tests = 10
        I_test = 0
        
        while True:
            t0 = time.time()
            epoc +=1
            errs.append(goalq - q_list[-1])
            errorq =  goalq - q_list[-1]
            errorqd = goal_qd - qd_list[-1] 
            # if len(errs) >=2 :
            #     errorqd += (errs[-1] - errs[-2])
            if np.sum(errorq ** 2) <= 0.0001 and epoc >= 100 :
                break 
            

            torque1 = q1_pid.Update(errorq[0],errorqd[0],self.system_time)
            torque2 = q2_pid.Update(errorq[1],errorqd[1],self.system_time)
            
            '''
                clculate force control
            '''
            I_test += np.array(errorq) * KI_tests * self.system_time
            qdd_d  = (np.array(errorq) * KP_test + np.array(errorqd) * KD_test ) + I_test
            input_torque = np.dot(self.robot.Mass(q_list[-1]),qdd_d) + self.robot.InverseDynamics(q_list[-1],qd_list[-1],[0,0])
            torque1 = q1_pid.Clip(input_torque[0])
            torque2 = q2_pid.Clip(input_torque[1])
            input_torque = [torque1,torque2]
            '''''' 
            # input_torque = [torque1,torque2] + torque_d 
            # torque1 = q1_pid.Clip(input_torque[0])
            # torque2 = q2_pid.Clip(input_torque[1])
            # input_torque = [torque1,torque2]
              
            if epoc % 10 == 0:
                print("input",np.round(input_torque,3),"errorq",np.round(errorq,3),"errorqd",np.round(errorqd,3))
            
            # qdd_u = self.robot.ForwardDynamics(input_torque,q_list[-1]+errorq ,qd_list[-1]+errorqd)
            # qdd_u = self.robot.ForwardDynamics(input_torque,errorq,errorqd)
            # qdd_u = self.robot.ForwardDynamics(input_torque,q_list[-1],qd_list[-1])
            # qdd_u = np.clip(qdd_u,-0.5,0.5)

            
            # q,qd = self.robot.Euler_Integral(q_list[-1],qd_list[-1],qdd_list[-1],self.system_time)
            q,qd,qdd_u = self.robot.Runge_Kutta4_Integral(q_list[-1],qd_list[-1],input_torque,self.system_time)
            
            qdd_list.append(qdd_u)
            q_list.append(q)
            qd_list.append(qd)

            
            Pos = self.robot.Fk(q_list[-1])
            self.robot.Fixed_Src(Pos)
            self.Draw()
            
            q_tmp = copy.deepcopy(q_list[-1]) 
            
            plotq1.append(radTdeg(q_tmp[0]))
            plotq2.append(radTdeg(q_tmp[1]))
            ts.append(epoc * self.system_time)
            
            if epoc > max_epoc:
                break
            t1 = time.time()
            ct = t1 - t0
            delay = self.system_time - ct
            if delay > 0: 
                time.sleep(delay)
                  
        print("end",self.robot.end)        
        fig1 = plt.figure(figsize=((12,8)))
        fig2 = plt.figure(figsize=((12,8)))
        ax = fig1.add_subplot(1,1,1)
        ax2 = fig2.add_subplot(1,1,1)
        ax.plot(ts,plotq1,color ='r',label="q1")
        ax2.plot(ts,plotq2,color ='b',label="q2")
        ax.legend()
        ax2.legend()
        plt.show()
        
    def FreeFall(self,period = 1000):
        n = 2
        torque_initial = np.array([0,0])
        q_initial = np.array([0,0])
        qd_initial = np.array([0,0])
        q_list = []
        qd_list = []
        qdd_list = []
        q_list.append(q_initial)
        qd_list.append(qd_initial)
        qdd_initial = self.robot.ForwardDynamics(torque_initial,q_initial,qd_initial)
        qdd_list.append(qdd_initial) 
        plot_q1 = []
        plot_q2 = []
        while True:
            t0 = time.time()
            period -= 1
            q,qd,qdd = self.robot.Runge_Kutta4_Integral(q_list[-1],qd_list[-1],torque_initial,self.system_time)
            # q,qd = self.robot.Euler_Integral(q_list[-1],qd_list[-1],qdd_list[-1],self.system_time)
            q_list.append(q)
            qd_list.append(qd)

            # torque = self.robot.InverseDynamics(q_list[-1],qd_list[-1],qdd_list[-1])
            # qdd = self.robot.ForwardDynamics(torque_initial,q_list[-1],qd_list[-1])
            qdd_list.append(qdd)
            Pos = self.robot.Fk(q_list[-1])
            self.robot.Fixed_Src(Pos)
            self.Draw()
            
            q_tmp = copy.deepcopy(q_list[-1]) 
            plot_q1.append(q_tmp[0] % (2*pi))
            plot_q2.append(q_tmp[1] % (2*pi))
            

            if period == 0:
                break
            t1 = time.time()
            ct = t1 - t0
            delay = self.system_time - ct
            if delay > 0: 
                time.sleep(delay)
        
       
        fig = plt.figure(figsize=((12,8)))
        ax = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax.plot(plot_q1,color ='r')
        ax.set_title("q1")
        ax2.plot(plot_q2,color ='b')
        ax2.set_title("q2")
        ax.legend()
        plt.show()
            # td.GetCounter(1)
            # ct = td.Spend(0,1)
            # delay = self.robot.dt - ct
            # td.Delay(delay)

    def Draw(self):
        data = [self.robot.src,
                self.robot.p1,
                self.robot.end
            ]

        self.queuy_gl.put(data)

if __name__ =='__main__':
    # q = []
    # test = openGL_Widget(q)
    # test.start()
    # while(1):
    #     try:
    #         time.sleep(1)
    #     except KeyboardInterrupt:
    #         break
    
    q_in = np.array([0,0])
    qd_in = np.array([0,0])
    qdd_in = np.array([0,0])

    app = QApplication(sys.argv)
    w = RobotAPP()
    w.show()
    
    torque_in = np.array([0,0])
    goalq = [0.872,-0.855]
    goalqd = [0,0]
    # w.PID(goalq,goalqd,q_in,qd_in,torque_in)
    w.FreeFall()
    app.exec()
    print('exit form')

