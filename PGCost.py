import pickle
import numpy as np
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


train_file = pickle.load(open('./train_file.pickle', 'rb'))
test_file = pickle.load(open('./test_file.pickle', 'rb'))

x = []
y = []
for tf in train_file:
    data = json.load(open('./processed_data/' + tf, 'r'))
    x.append(data['Plan']['Total Cost'])
    if 'Execution Time' in data:
        if data['Execution Time'] < 1800000:
            y.append(data['Execution Time'])
        else:
            y.append(1800000)
    else:
        y.append(1800000)


x = np.array(x)
y = np.array(y)
x = x.reshape((-1,1))
y = y.reshape((-1,1))
model = LinearRegression()
model.fit(x,y)

test_x = []
test_y = []
error2 = []
for testf in test_file:
    data = json.load(open('./processed_data/' + testf, 'r'))
    test_x.append(data['Plan']['Total Cost'])
    if 'Execution Time' in data:
        if data['Execution Time'] < 1800000:
            test_y.append(data['Execution Time'])
        else:
            test_y.append(1800000)
    else:
        test_y.append(1800000)
    # error2.append(max(test_x[-1], test_y[-1])/min(test_x[-1], test_y[-1]))

test_x = np.array(test_x)
test_y = np.array(test_y)
test_x = test_x.reshape((-1,1))
_y = model.predict(test_x)
error = []
for i in range(len(test_x)):
    error.append(max(_y[i][0], test_y[i])/min(_y[i][0], test_y[i]))

error = np.array(error)
print('median: ', np.median(error), ' mean: ', np.mean(error), ' max: ', np.max(error), ' len:',len(error))
print('median: ', np.median(error2), ' mean: ', np.mean(error2), ' max: ', np.max(error2), ' len: ',len(error2))
np.save('./PGerror.npy', error)


