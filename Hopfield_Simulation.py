import numpy as np 
from Functions import *
import matplotlib as mpl
import matplotlib.pyplot as plt

exercise = 4.1

if exercise==1:
    M_val =np.geomspace(2,1000,1000)
    N1 = 100
    N2 = 1000
    error1 = error_probability(M=M_val,N=N1)
    error2 = error_probability(M=M_val,N=N2)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(M_val, error1, label="Theoretical network with 100 neurons")
    plt.plot(M_val, error2, label="Theoretical network with 1000 neurons")
    plt.axvline(0.1*N1, 0, 1,alpha = 0.5 ,color='r',linestyle='-.', label='M=10')
    plt.axvline(0.1*N2, 0, 1,alpha = 0.5 ,color='r',linestyle='-.', label='M=100')
    ax.set_xscale("log")
    ax.set_xlabel("Number of patterns stored - M")
    ax.set_ylabel("Error probability")
    plt.legend(prop={'size': 7})
    plt.show()

if exercise==2:
    M_val1 =np.geomspace(2,1000,1000)
    N = 100
    max_iter = 1000
    error_theoretical = error_probability(M=M_val1,N=N)
    M_val2=np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    error_practical = np.zeros(np.size(M_val2))
    error_array = np.zeros(50)
    
    for i in range(np.size(M_val2)):

        for k in range(50):
            error_array[k]=network_simulation_Q3(M=M_val2[i],N=N)
        error_practical[i]=np.mean(error_array)

    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(M_val1, error_theoretical, alpha=0.8, label="Theoretical Error Probability")
    plt.scatter(M_val2, error_practical,c='r',marker='x',label="Practical Error Probability")
    ax.set_xscale("log")
    ax.set_xlabel("Number of patterns stored - M")
    ax.set_ylabel("Error probability")
    plt.legend(prop={'size': 7})
    plt.show()

if exercise ==3:
    M_val=np.geomspace(2,1000,40)
    M_val1 =np.geomspace(2,1000,1000)
    N=100
    error_theoretical = error_probability(M=M_val1,N=N)
    
    p_val = np.array([0.01,0.02,0.05,0.1,0.2,0.5,0.8,0.9,1])
    print(M_val.shape[0])
    print(network_simulation_Q4(M=500,N=100,p=0.2))

    error_practical = np.zeros((p_val.shape[0],M_val.shape[0]))
    error_array = np.zeros(20)

    for i in range(p_val.shape[0]):
        for j in range(M_val.shape[0]):
            print(j)
            for k in range(20):
                error_array[k]=network_simulation_Q4(M=int(M_val[j]),N=N,p=p_val[i])
            error_practical[i,j]=np.mean(error_array)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(M_val, error_practical[0,:],marker='x',label="p =0.01")
    plt.scatter(M_val, error_practical[1,:],marker='x',label="p =0.02")
    plt.scatter(M_val, error_practical[2,:],marker='x',label="p =0.05")
    plt.scatter(M_val, error_practical[3,:],marker='x',label="p =0.1")
    plt.scatter(M_val, error_practical[4,:],marker='x',label="p =0.2")
    plt.scatter(M_val, error_practical[5,:],marker='x',label="p =0.5")
    plt.scatter(M_val, error_practical[6,:],marker='x',label="p =0.8")
    plt.scatter(M_val, error_practical[7,:],marker='x',label="p =0.9")
    plt.scatter(M_val, error_practical[8,:],marker='x',label="p =1.0")
    plt.plot(M_val1, error_theoretical, alpha=0.8, label="Theoretical Error Probability")
    ax.set_xscale("log")
    ax.set_xlabel("Number of patterns stored - M")
    ax.set_ylabel("Error probability")
    plt.legend(prop={'size': 7})
    plt.show()




