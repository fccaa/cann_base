#!/usr/bin/env python3
"""
Program:  cann_base.py
Author:   Chi Chung Alan Fung

This program is an implementation of the model used in the papers by Fung, 
Wong and Wu in 2008 and 2010. This program is written for those who are 
interested in the work. Also, this program code could be a reference program 
code for researchers who are going to work on the model.
"""

import numpy as np
import scipy.integrate as spint
import sys

class cann_model:
    z_min = - np.pi;            
    z_range = 2.0 * np.pi;
    tau = 1.0
        
    def dist(self, c):
        tmp = np.remainder(c, self.z_range)
        if isinstance(tmp, (int, float)):
            if tmp > (0.5 * self.z_range):
                return (tmp - self.z_range);
            return tmp;
        
        for tmp_1 in np.nditer(tmp, op_flags=['readwrite']):
            if tmp_1 > (0.5 * self.z_range):
                tmp_1[...] = tmp_1 - self.z_range;
        
        return tmp;
    
    
    def __init__(self, argument):
        self.k = argument.k;
        self.a = argument.a;
        self.N = argument.N;
        self.dx = self.z_range / self.N
        
        self.x = (np.arange(0,self.N,1)+0.5) * self.dx + self.z_min;
                
        self.Jxx = np.zeros((self.N, self.N));
        for i in range(self.Jxx.shape[0]):
            for j in range(self.Jxx.shape[1]):
                self.Jxx[i][j] = \
                np.exp(-0.5 * np.square(self.dist(self.x[i] - self.x[j]) / self.a)) \
                / (np.sqrt(2*np.pi) * self.a);
                
        self.u = np.zeros((self.N));
        self.r = np.zeros((self.N));
        self.input = np.zeros((self.N));
            
    def set_input(self, A, z0):
        self.input = \
        A * np.exp(-0.25 * np.square(self.dist(self.x - z0) / self.a));
        
    def cal_r_or_u(self, u):
        u0 = 0.5 * (u + np.abs(u));
        r = np.square(u0);
        B = 1.0 + 0.125 * self.k * np.sum(r) * self.dx / (np.sqrt(2*np.pi) * self.a);
        r = r / B;
        return r;
        
    def get_dudt(self, t, u):
        dudt = \
        -u + np.dot(self.Jxx, self.cal_r_or_u(u)) * self.dx + self.input;
        dudt = dudt / self.tau;
        return dudt
        
                
"""
Begining of the program
"""

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

cann.set_input(1,0)

out = spint.solve_ivp(cann.get_dudt, (0, 1000), cann.u, method="RK45");

cann.u = out.y[:,out.y.shape[1]-1]

cann.set_input(0,0)

out = spint.solve_ivp(cann.get_dudt, (0, 1000), cann.u, method="RK45");



