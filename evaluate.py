import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras import regularizers

kFold = 10
layerSize = 30
dropout = 0.05
l2 = 0.0001

X = np.load('X_binary_com.npy')
Y = np.load('Y_binary_com.npy')

ids = np.arange(len(X))
es = k.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=3)

losses = []
precs = []
recs = []
f1s = []
maes = []
for _ in range(kFold):
    print(_)
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
                              epochs=30, batch_size=32, callbacks=[es], verbose=0)

    losses.append(train_history.history['val_loss'][-1])
    maes.append(train_history.history['val_binary_accuracy'][-1])

    yHat = model.predict(X[test])
    # classification
    predHat = np.array(np.round(yHat[:,0], 0), dtype=int)
    pred = Y[test]
    # regression
    #predHat = np.array(yHat[:,1] > yHat[:,0], dtype=int)
    #pred = np.array(Y[test,1] > Y[test,0], dtype=int)

    prec, rec, f1, supp = precision_recall_fscore_support(pred, predHat)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)
    del model

losses = np.array(losses)
precs = np.array(precs)
recs = np.array(recs)
f1s = np.array(f1s)
print('loss', '###')
print('mean', np.mean(losses), 'std', np.std(losses))
print('mae', '###')
print('mean', np.mean(maes), 'std', np.std(maes))
print('prec', '###')
print('mean', np.mean(precs, axis=0), 'std', np.std(precs, axis=0))
print('rec', '###')
print('mean', np.mean(recs, axis=0), 'std', np.std(recs, axis=0))
print('f1', '###')
print('mean', np.mean(f1s, axis=0), 'std', np.std(f1s, axis=0))


### Non commulative regression
# loss ###
# mean 1.875868750990508 std 0.3036245856098755
# accuracy ###
# mean 0.704553892215569 std 0.1329495314285091
