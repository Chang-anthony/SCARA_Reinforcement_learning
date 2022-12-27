#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_runing_avg_learning_curve(x, scores,title, figure_file):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    ax.plot(x, running_avg)
    ax.set_title(title)
    fig.savefig(figure_file)
    
def plot_learning_curve(x, scores,title, figure_file):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, scores)
    ax.set_title(title)
    fig.savefig(figure_file)