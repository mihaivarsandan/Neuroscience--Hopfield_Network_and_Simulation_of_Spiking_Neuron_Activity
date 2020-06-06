import numpy as np 
from Functions import *
import matplotlib as mpl
import matplotlib.pyplot as plt

exercise=1.4

if exercise==1.1:
    V = np.array([i for i in range(0, 200, 1)]) - 100
    am = alpha_m(V)
    bm = beta_m(V)
    ah = alpha_h(V)
    bh = beta_h(V)
    an = alpha_n(V)
    bn = beta_n(V)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(V,am,label=r'$\alpha_m$')
    ax1.plot(V,bm,label=r'$\beta_m$')
    ax1.legend(prop={'size': 12},loc=2)
    ax1.set_xlabel("Action Potential mV")
    ax1.set_ylabel("Activation Level")

    ax2.plot(V,ah,label=r'$\alpha_h$')
    ax2.plot(V,bh,label=r'$\beta_h$')
    ax2.legend(prop={'size': 12},loc=2)
    ax2.set_xlabel("Action Potential mV")
    ax2.set_ylabel("Activation Level")

    ax3.plot(V,an,label=r'$\alpha_n$')
    ax3.plot(V,bn,label=r'$\beta_n$')
    ax3.legend(prop={'size': 12},loc=2)
    ax3.set_xlabel("Action Potential mV")
    ax3.set_ylabel("Activation Level")
    #plt.subplots_adjust(hspace=0.5)
    plt.show()

if exercise==1.2:
    T=200
    A=6.3
    Pulse = I_pulse(T,A)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(np.array([0.001*i for i in range(Pulse.shape[0])]),Pulse, label="Input Puse")
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Amplitude mA/nF")
    plt.show()
    """
    t, V, m, h, n = neuron_simulation(Pulse)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(t,V, label="Neuron Membrane Voltage")
    ax.set_xlabel("Time, ms",fontsize=16)
    ax.set_ylabel('Membrane Voltage, mV',fontsize=16)
    plt.show()

if exercise==1.3:
    T=10
    p=5
    I=-2.3
    Pulse = I_pulse_periodic(T=T,p=p,I=I)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(np.array([0.001*i for i in range(Pulse.shape[0])]),Pulse, label="Input Puse")
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Amplitude mA/nF")
    plt.show()

if exercise==1.4:
    T=18
    p=5
    I=-4
    Pulse = I_pulse_periodic(T=T,p=p,I=I)

    t, V, m, h, n = neuron_simulation(Pulse)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(t,V, label="Neuron Membrane Voltage")
    ax.set_xlabel("Time, ms",fontsize=16)
    ax.set_ylabel('Membrane Voltage, mV',fontsize=16)
    plt.show()
    