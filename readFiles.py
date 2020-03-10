from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def _listToPaddedArray(list):
    maxLen = max([len(item) for item in list])
    result = np.zeros((len(list), maxLen))
    for i, item in enumerate(list):
        result[i, :len(item)] = item
    return result


data = loadmat('../platoon/MonteCarlo1000Second.mat')

NS = int(data['NS'][0])

N_t = np.array(data['N_t'][0])

N_platoon = []
for i in data['N_p'][0]:
    l = []
    for j in i[0]:
        l.append(j)
    N_platoon.append(l)
N_platoon = np.array(N_platoon)
N_platoon[:, :, :, 0] = np.round(N_platoon[:, :, :, 0], 0)


N_single = []
for i in data['N_single'][0]:
    l = []
    for j in i[0]:
        l.append(j)
    N_single.append(l)
N_single = np.array(N_single)
N_single[:, :, :, 0] = np.round(N_single[:, :, :, 0], 0)


V_t = []
for v in data['V_t'][0]:
    V_t.append(np.array(v)[0])
V_t = _listToPaddedArray(V_t)

S_t = []
for s in data['S_t'][0]:
    S_t.append(np.array(s)[0])
S_t = _listToPaddedArray(S_t)

L_p = data['L_p'][0]

tabularX = np.hstack([N_t.reshape([-1, 1]),
                      L_p.reshape([-1, 1])])

X = []
Y = []
for i in range(NS):
    print(i)
    for loc in range(N_single.shape[1]):
        for time in range(N_single.shape[2]):
            n_s = N_single[i, loc, time, 0]
            n_s_prev = N_single[i, loc, max(0,time-1), 0]
            n_p = N_platoon[i, loc, time, 0]
            n_p_prev = N_platoon[i, loc, max(0,time-1), 0]
            if n_s != np.inf and n_p != np.inf and n_s_prev != np.inf and n_p_prev != np.inf:
                X.append(np.concatenate([np.array([N_t[i]]), np.array([L_p[i]]), np.array([loc]), np.array([time]), V_t[i], S_t[i]]))
                #Y.append(np.array([n_s - n_s_prev, n_p - n_p_prev])) # non commulative over time
                #Y.append(1) if n_s - n_s_prev < n_p - n_p_prev else Y.append(0)
                Y.append(1) if n_s < n_p else Y.append(0)

X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

np.save("X_binary_com.npy", X)
np.save("Y_binary_com.npy", Y)
