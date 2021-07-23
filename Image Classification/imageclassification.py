import sys
from mxnet import gluon,autograd,np,npx
import mxnet as d21
#import pygame as d21
import matplotlib.pyplot as plt
import time
from IPython import display

npx.set_np()
batch_size = 256

#d21.use_svg_display()

mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)

#print(len(mnist_train), len(mnist_test))
#print(mnist_train[0][0].shape)

class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def get_fashion_mnist_labels(labels):
    #return text labels for the dataset)
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols,titles=None,scale=1.5):
    figsize = (num_cols*scale,num_rows*scale)
    _,axes = plt.subplots(num_rows,num_cols,figsize=figsize)
    axes = axes.flatten()
    for i, (ax,img) in enumerate(zip(axes,imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X,y = mnist_train[:18]
#print(X.shape)
#show_images(X.squeeze(axis=-1),2,8,titles=get_fashion_mnist_labels(y))
#plt.show()



def get_dataloader_workers():

    return 0 if sys.platform.startswith('win') else 4

transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),batch_size, shuffle= True, num_workers =get_dataloader_workers())

#tune to read training data
"""
print(time.perf_counter())
for X,y in train_iter:
    continue
print(time.perf_counter())
"""

def load_data_fashion_mnist(batch_size, resize=None):
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0,dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers()), gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))


"""
load data
train_iter,test_iter = load_data_fashion_mnist(32,resize=64)
for X,y in train_iter:
    print(X.shape,X.dtype, y.shape, y.dtype)
    break
"""

train_iter,test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = np.random.normal(0,0.01,(num_inputs,num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()

def softmax(X):
    X_exp = np.exp(X)
    X_exp_normalization = X_exp.sum(1,keepdims=True) # denominator or normalization constant or partition function
    return X_exp / X_exp_normalization

#X= np.random.normal(0,1,(2,5))
#print(X)
#X_prob  = softmax(X)
#print(X_prob, X_prob.sum(1))

y = np.array([0, 2])
y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]

""" Flattens each original image in the batch into a vector"""
def net(X):
    #print(W.shape[0])
    return softmax(np.dot(X.reshape((-1,W.shape[0])),W)+b)

def cross_entropy(y_hat,y):
    return -np.log(y_hat[range(len(y_hat)),y])

def accuracy(y_hat,y):

    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.astype(y.dtype)==y
    return float(cmp.astype(y.dtype).sum())

#print(accuracy(y_hat, y) / len(y))

def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]
print(evaluate_accuracy(net,test_iter))


def train_epoch_ech3(net, train_iter, loss, updater):
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X,y in train_iter:
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat,y)
        l.backward()
        update(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat,y),y.size)
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
