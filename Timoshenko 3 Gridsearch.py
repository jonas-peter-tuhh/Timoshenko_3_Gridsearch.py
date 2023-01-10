# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import sys
import os
#insert path of parent folder to import helpers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = True

class Net(nn.Module):
    def __init__(self, num_layers, layers_size):
        super(Net, self).__init__()
        assert num_layers == len(layers_size)
        self.linears = nn.ModuleList([nn.Linear(1, layers_size[0])])
        self.linears.extend([nn.Linear(layers_size[i-1], layers_size[i])
                            for i in range(1, num_layers)])
        self.linears.append(nn.Linear(layers_size[-1], 1))

    def forward(self, x) :
        x = torch.unsqueeze(x, 1)
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output
##
# Hyperparameter
learning_rate = 0.01
# Definition der Parameter des statischen Ersatzsystems

# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
EI = 21
K = 5/6
G = 80
A = 100

#Normierungsfaktor (siehe Kapitel 10.3)
normfactor = 10/((11*Lb**5)/(120*EI))

# ODE als Loss-Funktion, Streckenlast
Ln = 0 #float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
Lq = Lb # float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
s = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')


def h(x):
    return eval(s)

#Netzwerk für Biegung
def f(x, net_B):
    u = net_B(x)
    _, _, _, u_xxxx = deriv(u, x, 4)
    ode = u_xxxx + (h(x - Ln))/EI
    return ode

#Netzwerk für Schub
def g(x, net_S):
    u = net_S(x)
    _, u_xx = deriv(u, x, 2)
    ode = u_xx - (h(x - Ln))/ (K * A * G)
    return ode


x = np.linspace(0, Lb, 1000)
pt_x = myconverter(x)
qx = h(x)* (x <= (Ln + Lq)) * (x >= Ln)


Q0 = integrate.cumtrapz(qx, x, initial=0)

qxx = (qx) * x

M0 = integrate.cumtrapz(qxx, x, initial=0)
##
def gridSearch(num_layers, layers_size):
    start = time.time()
    net_B = Net(num_layers, layers_size)
    net_S = Net(num_layers, layers_size)
    net_B = net_B.to(device)
    net_S = net_S.to(device)
    mse_cost_function = torch.nn.MSELoss()  # Mean squared error
    optimizer = torch.optim.Adam([{'params': net_B.parameters()}, {'params': net_S.parameters()}], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.75)
    if train:
        y1 = net_B(torch.unsqueeze(myconverter(x, False), 1)) + net_S(torch.unsqueeze(myconverter(x, False), 1))
        fig = plt.figure()
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-10, 0])
        net_out_plot = myconverter(y1)
        line1, = ax1.plot(x, net_out_plot)
        plt.title(f'{num_layers =}, {layers_size =}')
        plt.show(block=False)
        pt_x = torch.unsqueeze(myconverter(x), 1)
        f_anal = (-1 / 120 * normfactor * x ** 5 + 1 / 6 * Q0[-1] * x ** 3 - M0[-1] / 2 * x ** 2) / EI + (
                1 / 6 * normfactor * x ** 3 - Q0[-1] * x) / (K * A * G)

    ##
    iterations = 100000
    for epoch in range(iterations):
        if not train: break
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = myconverter(x_bc)

        x_collocation = np.random.uniform(low=0.0, high=Lb, size=(250 * int(Lb), 1))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = myconverter(x_collocation)
        pt_all_zeros = myconverter(all_zeros, False)
        f_out_B = f(pt_x_collocation, net_B)
        f_out_S = g(pt_x_collocation, net_S)

        # Randbedingungen
        net_bc_out_B = net_B(pt_x_bc)
        net_bc_out_S = net_S(pt_x_bc)
        vb_x, vb_xx, vb_xxx = deriv(net_bc_out_B, pt_x_bc, 3)
        vs_x = deriv(net_bc_out_S, pt_x_bc, 1)

        # RB für Biegung
        BC3 = net_bc_out_B[0]
        BC6 = vb_xxx[0] - Q0[-1] / EI
        BC7 = vb_xxx[-1]
        BC8 = vb_xx[0] + M0[-1] / EI
        BC9 = vb_xx[-1]
        BC10 = vb_x[0]

        # RB für Schub
        BC2 = net_bc_out_S[0]
        BC4 = vs_x[0] + Q0[-1] / (K * A * G)
        BC5 = vs_x[-1]

        mse_Gamma_B = errsum(mse_cost_function, BC3, 1 / normfactor * BC6, BC7, 1 / normfactor * BC8, BC9, BC10)
        mse_Gamma_S = errsum(mse_cost_function, BC2, 1 / normfactor * BC4, BC5)
        mse_Omega_B = errsum(mse_cost_function, f_out_B)
        mse_Omega_S = errsum(mse_cost_function, f_out_S)

        loss = mse_Gamma_B + mse_Gamma_S + mse_Omega_B + mse_Omega_S

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Training Loss:", loss.data)
                plt.grid()
                net_out = myconverter(net_B(pt_x) + net_S(pt_x))
                err = np.linalg.norm(net_out - f_anal, 2)
                print(f'Error = {err}')
                if err < 0.1 * Lb:
                    print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    break
                line1.set_ydata(net_out)
                fig.canvas.draw()
                fig.canvas.flush_events()

# GridSearch
time_elapsed = []
for num_layers in range(2, 4):
    for _ in range(10):  # Wieviele zufällige layers_size pro num_layers
        layers_size = [np.random.randint(30, 200) for _ in range(num_layers)]
        time_elapsed.append(
            (num_layers, layers_size, gridSearch(num_layers, layers_size)))
        plt.close()

with open(r'random14m2.3s.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')
##
