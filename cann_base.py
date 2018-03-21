#!/usr/bin/env python3
"""
Program:  cann_base.py
Author:   Chi Chung Alan Fung

This program is an implementation of the model used in the papers by Fung, 
Wong and Wu in 2008 and 2010. This program is written for those who are 
interested in the work. Also, this program code could be a reference program 
code for researchers who are going to work on the model.
"""
from numba import jit

import numpy as np
import scipy as sp

class cann_model:
    z_min = - np.pi
    z_range = 2 * np.pi
    def __init__(self, argument):
        self.k = argument.k;
        self.a = argument.a;
        self.N = argument.N;
        self.x = (np.arange(0,self.N,1)+0.5) * self.z_range / self.N + self.z_min;
        self.Jxx = np.zeros((self.N, self.N));
        for i in range(self.Jxx.shape[0]):
            for j in range(self.Jxx.shape[1]):
                tmp = np.exp2(-0.5 * (self.x[i] - self.x[j]) / self.a) \
                /(np.sqrt(2*np.pi) * self.a)
                self.Jxx[i][j] = tmp
                
        
                    

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-k", metavar="float", type=float, \
                    help="rescaled Inhibition", default=0.5)
parser.add_argument("-a", metavar="float", type=float, \
                    help="width of excitatory couplings", default=0.5)
parser.add_argument("-N", metavar="int", type=int, \
                    help="number of excitatory units", default=128)
arg = parser.parse_args()

cann = cann_model(arg)

print(cann.x)
