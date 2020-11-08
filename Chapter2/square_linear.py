import torch
torch.__version__

import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
import math


# Training Data
def get_data():
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype),requires_grad=False).view(17,1)
    y = Variable(torch.from_numpy(train_Y).type(dtype),requires_grad=False)
    return X,y

def plot_variable(x,y,z='',**kwargs):
    plt.plot(x.data.numpy(), y.data.numpy(), z, **kwargs)

def get_weights():
    w = Variable(torch.randn(1),requires_grad = True)
    b = Variable(torch.randn(1),requires_grad=True)
    c = Variable(torch.randn(1),requires_grad=True)
    return w,b,c

def simple_network(x):
    x2 = torch.mul(x,x)
    y_pred = torch.matmul(x2,w)+torch.matmul(x,b)+c
    return y_pred

def loss_fn(y,y_pred):
    loss = (y_pred-y).pow(2).sum()
    for param in [w,b]:
        if not param.grad is None: param.grad.data.zero_()
    # breakpoint()
    loss.backward()
    return loss.data.item()


def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    if math.isnan(w.grad.data) or math.isinf(w.grad.data):
        breakpoint()
    b.data -= learning_rate * b.grad.data
    c.data -= learning_rate * c.grad.data

learning_rate = 1e-5

x,y = get_data()               # x - represents training data,y - represents target variables
w,b,c = get_weights()           # w,b - Learnable parameters
for i in range(50000):
    y_pred = simple_network(x) # function which computes wx + b
    loss = loss_fn(y,y_pred)   # calculates sum of the squared differences of y and y_pred
    if i % 500 == 0:
        print(loss)
    optimize(learning_rate)    # Adjust w,b to minimize the loss

x_numpy = x.data.numpy()
plot_variable(x,y,'ro')
plot_variable(x,y_pred,label='Fitted line')
plt.show()
print(w,b,c)