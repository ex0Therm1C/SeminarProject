import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import gc

X = np.load('X_binary_com.npy')
Y = np.load('Y_binary_com.npy')

ids = np.arange(len(X))
es = k.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=1)

dropouts = [0.05, 0.1, 0.2]
l2Regs = [0.00005, 0.0001, 0.0002]
layerSizes = [20, 25, 30]
kFold = 2

bestLoss = 100000
bestConfig = None

def trainModel(X, Y, dropout, l2, layerSize):
    valLoss = 0
    for _ in range(kFold):
        np.random.shuffle(ids)
        split = int(len(X) * 0.8)
        train = ids[:split]
        test = ids[split:]

        model = k.Sequential([
            k.layers.Input(X.shape[1]),
            k.layers.Dense(layerSize * 2, activation='relu', kernel_regularizer=regularizers.l2(l2)),
            k.layers.Dropout(dropout),
            k.layers.Dense(layerSize, activation='relu', kernel_regularizer=regularizers.l2(l2)),
            k.layers.BatchNormalization(),
            k.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(k.optimizers.Adam(), k.losses.binary_crossentropy, metrics=[k.metrics.binary_accuracy])
        train_history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]),
                                  epochs=8, batch_size=32, callbacks=[es], verbose=0)
        valLoss += train_history.history['val_loss'][-1]
        del model
        gc.collect()
    valLoss /= kFold
    return valLoss


def forwardSearch():
    unusedVars = list(np.arange(X.shape[1]))
    usedVars = []
    valError = np.inf

    while True:
        minErr = np.inf
        minVar = None
        for testedVar in unusedVars:
            newSubset = usedVars + [testedVar]
            err = trainModel(X[:, newSubset], Y, 0.05, 0.0001, 30)
            print(testedVar)
            if err < minErr:
                minErr = err
                minVar = testedVar

        if minErr < valError:
            valError = minErr
            usedVars += [minVar]
            unusedVars = [v for v in unusedVars if v is not minVar]
        else:
            break
        if len(usedVars) == 2:
            break

    print('run ', len(usedVars), '|', usedVars)
    return usedVars


def runFS():
    vars = []
    for i in range(2):
        print('#########', i, '############')
        vars.append(forwardSearch())

    occurences = {}
    position = {}
    for usedVars in vars:
        for i in range(len(usedVars)):
            v = usedVars[i]
            if v not in occurences:
                occurences[v] = 0
            occurences[v] += 1
            if v not in position:
                position[v] = list()
            position[v].append(i)

    for v in range(X.shape[1]):
        if v not in occurences:
            print(v, 0)
        else:
            print(v, occurences[v], 'avrg pos', sum(position[v]) / float(occurences[v]))

runFS()


def gridSearch():
    for dropout in dropouts:
        for l2 in l2Regs:
            for layerSize in layerSizes:
                valLoss = trainModel(dropout, l2, layerSize)
                print('dropout', dropout, 'l2', l2, 'layerSize', layerSize, 'valLoss', valLoss)
                if valLoss < bestLoss:
                    bestLoss = valLoss
                    bestConfig = (dropout, l2, layerSize)

    print('\n', 'xxxxxxxxxxxxxxxxxxxxxxx')
    print('best config')
    print('dropout', bestConfig[0], 'l2', bestConfig[1], 'layerSize', bestConfig[2], 'valLoss', bestLoss)


########### not comulative
# regression
# dropout 0.05 l2 0.0001 layerSize 30 valLoss 1.7063

# classification
# dropout 0.05 l2 0.0001 layerSize 30 valLoss 0.07517569169446736

########## commulative
# regression
# dropout 0.05 l2 0.0002 layerSize 30 valLoss 3.5053

# classification
# dropout 0.05 l2 0.0001 layerSize 30 valLoss 0.07525175895318835