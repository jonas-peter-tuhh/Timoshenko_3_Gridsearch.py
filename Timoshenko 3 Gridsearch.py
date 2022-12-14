# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import scipy.integrate
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import math

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

    def forward(self, x):  # ,p,px):
        # torch.cat([x,p,px],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        inputs = x
        for i in range(len(self.linears)-1):
            x = torch.tanh(self.linears[i](x))
        output = self.linears[-1](x)
        return output
##
choice_load = input("Möchtest du ein State_Dict laden? y/n")
if choice_load == 'y':
    train=False
    filename = input("Welches State_Dict möchtest du laden?")
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load('C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko NN Kragarm 5.3\\saved_data\\'+filename))
    net.eval()
##
# Hyperparameter
learning_rate = 0.0075

# Definition der Parameter des statischen Ersatzsystems

Lb = float(input('Länge des Kragarms [m]: '))
E = 21#float(input('E-Modul des Balkens [10^6 kNcm²]: '))
h = 10#float(input('Querschnittshöhe des Balkens [cm]: '))
b = 10#float(input('Querschnittsbreite des Balkens [cm]: '))
A = h*b
I = (b*h**3)/12
EI = E*I*10**-3
G = 80#float(input('Schubmodul des Balkens [GPa]: '))
LFS = 1#int(input('Anzahl Streckenlasten: '))
K = 5 / 6  # float(input(' Schubkoeffizient '))
Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS
normfactor = 10/(Lb**3/(K*A*G)+(11*Lb**5)/(120*EI))


for i in range(LFS):
    # ODE als Loss-Funktion, Streckenlast
    Ln[i] = 0#float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
    Lq[i] = Lb#float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
    s[i] = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')

def h(x, j):
    return eval(s[j])

#Netzwerk für Biegung
def f(x, net_B):
    u = net_B(x)
    u_x = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xxx = torch.autograd.grad(u_xx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xxxx = torch.autograd.grad(u_xxx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    ode = 0
    for i in range(LFS):
        #0 = vb'''' + q(x)/EI
        ode += u_xxxx + h(x - Ln[i], i) / EI * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

#Netzwerk für Schub
def g(x, net_S):
    u = net_S(x)
    u_x = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    #0 = vs'' - q(x)/KAG
    ode = u_xx - h(x - Ln[i], i) / (K * A * G) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode


x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)
qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1) - Ln[i], i).cpu().detach().numpy()).squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)
#Q0 = Q(0) = int(q(x)), über den ganzen Balken
qxx = qx * x
#M0 = M(0) = int(q(x)*x), über den ganzen Balken
M0 = integrate.cumtrapz(qxx, x, initial=0)
#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten

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
        # + net_S(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
        y1 = net_S(torch.unsqueeze(Variable(torch.from_numpy(
            x).float(), requires_grad=False).to(device), 1)) +\
            net_B(torch.unsqueeze(Variable(torch.from_numpy(
            x).float(), requires_grad=False).to(device), 1))
        fig = plt.figure()
        plt.title(f'{num_layers =}, {layers_size =}')
        plt.grid()
        ax1 = fig.add_subplot()
        ax1.set_xlim([0, Lb])
        ax1.set_ylim([-20, 0])
        # ax2.set_
        net_out_plot = y1.cpu().detach().numpy()
        line1, = ax1.plot(x, net_out_plot)
        plt.show(block=False)
        f_anal=(-1/120 * normfactor * pt_x**5 + 1/6 * Q0[-1] * pt_x**3 - M0[-1]/2 *pt_x**2)/EI + (1/6 * normfactor * (pt_x)**3 - Q0[-1]*pt_x)/(K*A*G)

    iterations = 1000000
    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(
            x_bc).float(), requires_grad=True).to(device), 1)
        # unsqueeze wegen Kompatibilität
        pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(),
                           requires_grad=False).to(device)

        x_collocation = np.random.uniform(
            low=0.0, high=Lb, size=(250 * int(Lb), 1))
        #x_collocation = np.linspace(0, Lb, 1000*int(Lb))
        all_zeros = np.zeros((250 * int(Lb), 1))

        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
        ode_B = f(pt_x_collocation, net_B)
        ode_S = g(pt_x_collocation, net_S)

        # Randbedingungen
        net_bc_out_B = net_B(pt_x_bc)
        net_bc_out_S = net_S(pt_x_bc)
        # ei --> Werte, die minimiert werden müssen
        u_x_B = torch.autograd.grad(net_bc_out_B, pt_x_bc, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(net_bc_out_B))[0]
        u_xx_B = torch.autograd.grad(u_x_B, pt_x_bc, create_graph=True, retain_graph=True,
                                     grad_outputs=torch.ones_like(net_bc_out_B))[0]
        u_xxx_B = torch.autograd.grad(u_xx_B, pt_x_bc, create_graph=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(net_bc_out_B))[0]
        u_x_S = torch.autograd.grad(net_bc_out_S, pt_x_bc, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(net_bc_out_S))[0]

        # RB für Biegung
        e1_B = net_bc_out_B[0]
        e2_B = u_x_B[0]
        e3_B = u_xxx_B[0] - Q0[-1] / EI
        e4_B = u_xx_B[0] + M0[-1] / EI
        e5_B = u_xxx_B[-1]
        e6_B = u_xx_B[-1]

        # RB für Schub
        e1_S = net_bc_out_S[0]
        e2_S = u_x_S[0] + Q0[-1] / (K * A * G)
        e3_S = u_x_S[-1]

        # Alle e's werden gegen 0-Vektor (pt_zero) optimiert.

        mse_bc_B = mse_cost_function(e1_B, pt_zero) + mse_cost_function(e2_B,
                                                                        pt_zero) + 1 / normfactor * mse_cost_function(
            e3_B, pt_zero) + 1 / normfactor * mse_cost_function(e4_B, pt_zero) + mse_cost_function(e5_B,
                                                                                                   pt_zero) + mse_cost_function(
            e6_B, pt_zero)
        mse_ode_B = 1 / normfactor * mse_cost_function(ode_B, pt_all_zeros)
        mse_bc_S = mse_cost_function(e1_S, pt_zero) + 1 / normfactor * mse_cost_function(e2_S,
                                                                                         pt_zero) + mse_cost_function(
            e3_S, pt_zero)
        mse_ode_S = 1 / normfactor * mse_cost_function(ode_S, pt_all_zeros)

        loss = mse_ode_S + mse_ode_B + mse_bc_S + mse_bc_B
        loss = 1 / normfactor * loss
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out = net_B(pt_x) + net_S(pt_x)
                err = torch.norm(net_out - f_anal, 2)
                print(f'Error = {err}')
                if err < 0.1 * Lb:
                    print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    break
                line1.set_ydata(net_B(
                    torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device),
                                    1)).cpu().detach().numpy() + net_S(
                    torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device),
                                    1)).cpu().detach().numpy())
                fig.canvas.draw()
                fig.canvas.flush_events()

# GridSearch
time_elapsed = []
for num_layers in range(2, 5):
    for _ in range(20):  # Wieviele zufällige layers_size pro num_layers
        layers_size = [np.random.randint(8, 15) for _ in range(num_layers)]
        time_elapsed.append(
            (num_layers, layers_size, gridSearch(num_layers, layers_size)))
        plt.close()

with open(r'timing2.txt', 'w') as fp:
    for item in time_elapsed:
        # write each item on a new line
        fp.write(f'{item} \n')
##
if choice_load == 'n':
    choice_save = input("Möchtest du die Netzwerkparameter abspeichern? y/n")
    if choice_save == 'y':
        filename = input("Wie soll das State_Dict heißen?")
        torch.save(net.state_dict(),'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko NN Kragarm 5.3\\saved_data\\'+filename)
##
