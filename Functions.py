import numpy as np 
from scipy.stats import norm, bernoulli
import random

def error_probability(M,N):
    error = 1- norm.cdf(np.sqrt((N-1)/(M-1)*0.5))
    return error

def create_network(memories):
    N = memories.shape[1]
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            W[i, j] = np.sum((memories[:, i] - 0.5)*(memories[:, j] - 0.5), axis=0)
    W = W + W.T
    return W

def F(val):
    if val >= 0:
        return 1
    else:
        return 0

def get_number_state_changes(init,W,N):
    new_state = np.zeros(np.shape(init))
    for i in range(N):
        next_bit = F(np.dot(W[i,:],init))
        new_state[i]=next_bit  
    return np.sum(new_state != init)

def network_simulation_Q3(M,N):
    memories= np.random.randint(2,size=(M,N))
    W = create_network(memories)
    value = 0
    for i in range(M):
        value += get_number_state_changes(memories[i,:],W,N)
    err_prob = value/(M*N)
    return err_prob

def flip_stored_patterns(patterns,p):
    new_patterns=np.zeros(np.shape(patterns))
    for i in range(patterns.shape[0]):
        for j in range(patterns.shape[1]):
            flip = np.random.binomial(1,p)
            if flip:
                new_patterns[i,j]=1 - patterns[i, j]
            else:
                new_patterns[i,j]=patterns[i,j]
    return new_patterns

def state_evolution(init,W,N):
    n = 0
    i = 0 
    next_state = init
    while n < N:
        next_bit = F(np.dot(W[i,:],init))
        if next_state[i] == next_bit:
            n += 1
        else:
            n = 0 
        next_state[i] = next_bit
        i = (i+1) % N 
    return next_state

def network_simulation_Q4(M,N,p):
    memories = np.random.randint(2,size=(M,N))
    W = create_network(memories)
    inital_events =flip_stored_patterns(memories,p)
    errors = 0
    for i in range(M):
        final_state=state_evolution(inital_events[i,:],W,N)
        errors += np.sum(final_state!=memories[i,:])
    err_prob = errors/(M*N)
    return err_prob

def alpha_m(V):
    return (2.5-0.1*(V+65))/(np.exp(2.5-0.1*(V+65))-1)
    
def beta_m(V):
    return 4*np.exp(-(V+65)/18)
    
def alpha_h(V):
    return 0.07*np.exp(-(V+65)/20)

def beta_h(V):
    return 1/(np.exp(3-0.1*(V+65))+1)

def alpha_n(V):
    return (0.1-0.01*(V+65))/(np.exp(1-0.1*(V+65))-1)

def beta_n(V):
    return 0.125*np.exp(-(V+65)/80)

def ss_var(alpha,beta):
    return alpha/(alpha+beta)

def neuron_simulation(I_ext):
    """Initial Parameters"""
    g_na = 120
    g_k = 36
    g_l = 0.3
    e_na = 50
    e_k = -77
    e_l = -54.4
    dt = 0.001
    V_0 = -65

    N = I_ext.shape[0]

    am = alpha_m(V_0)
    bm = beta_m(V_0)
    ah = alpha_h(V_0)
    bh = beta_h(V_0)
    an = alpha_n(V_0)
    bn = beta_n(V_0)
    
    m_0 = ss_var(am, bm)
    h_0 = ss_var(ah, bh)
    n_0 = ss_var(an, bn)
    """END"""

    V = np.zeros(N)
    m = np.zeros(N)
    h = np.zeros(N)
    n = np.zeros(N)

    V[0] = V_0
    m[0] = m_0
    h[0] = h_0
    n[0] = n_0

    t = np.array([dt * i for i in range(N)])


    for k in range(N-1):
        
        V_prev = V[k]
        m_prev = m[k]
        n_prev = n[k]
        h_prev = h[k]

        am = alpha_m(V[k])
        bm = beta_m(V[k])
        ah = alpha_h(V[k])
        bh = beta_h(V[k])
        an = alpha_n(V[k])
        bn = beta_n(V[k])

        V[k+1] = V[k]+dt*(-g_na * (m[k]**3)*h[k]*(V[k] - e_na) - g_k * (n[k]**4)*(V[k] - e_k) - g_l*(V[k] - e_l) + I_ext[k])
        m[k+1] = m[k]+dt*(am * (1-m[k]) - bm * m[k])
        h[k+1] = h[k]+dt*(ah * (1-h[k]) - bh * h[k])
        n[k+1] = n[k]+dt*(an * (1-n[k]) - bn * n[k])
    
    return t, V, m, h, n

def I_pulse(p, amplitude):
    dt = 0.001 # 0.001 ms
    timesteps = int(p/dt) # number of timesteps that need to be generated 

    pulse = np.ones(timesteps)*amplitude
    return pulse

def I_pulse_periodic(T,p,I):
    dt=0.001
    div,rem = divmod(200,T)
    pulse=np.zeros(int(200/dt))
    array=np.zeros(int(T/dt))
    array[int((T-p)/dt):int(T/dt)]=I*np.ones(int(p/dt))
    for num in range(div):
        pulse[int(T*num/dt):int(T*(num+1)/dt)]=array
    print(num)
    pulse[int(T*(num+1)/dt):]=array[:int(rem/dt)]
    return pulse

    


    
                






    

