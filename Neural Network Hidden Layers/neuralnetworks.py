
import numpy as np
import matplotlib.pyplot as plt


#init all trainable parameters randomly

def initialize_parameters(n_x, n_h):

    #n_x: number of input features
    #n_h: number of hidden units in the single hidden layer

    #return
    #W1: shape(n_h , n_x)
    #B1: shape(n_h, 1)
    #W2: shape(1, n_h)
    #B2: shape(1,1)

    np.random.seed(0)

    W1 = np.array(0.01 * np.random.randn(n_h, n_x))
    W2 = np.array(0.01 * np.random.randn(1, n_h))
    B1 = np.zeros((n_h,1))
    B2 = np.zeros((1,1))

    return W1,B1,W2,B2

#forward propagation
def forward_propagation(X, W1, B1, W2, B2):

    Z1 = np.matmul(W1,X)+B1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2,A1)+B2
    A2 = sigmoid(Z2)

    return A1,A2

def compute_cost(A2,Y):

    coeff = -1/len(Y)
    term1 = np.sum(Y*np.log(A2))
    term2 = np.sum((1-Y)*np.log(1-A2))
    cost = coeff*(term1+term2)

    return cost

def backward_propagation(X,Y,W2,A1,A2):
    """ Vectorized computation of the gradients needed for Gradient Descent
        on the trained parameters.

    Inputs:
        X:  NumPy array of input samples of shape (n, m)
        Y:  NumPy array (m,) with the known labels
        W2: NumPy array of second layer of parameters with shape (1, n_h)
        A1: NumPy array of shape (n_h, m) with the activations of the first hidden layer
        A2: NumPy array of shape (1, m) with the activations of the output layer

    Returns:
        dW1: NumPy array of shape (n_h, n_x)
        dB1: NumPy array of shape (n_h, 1)
        dW2: NumPy array of shape (1, n_h)
        dB2: NumPy array of shape (1, 1)
    """

    coeff = 1/len(Y)
    dZ2 = A2-Y
    Z1  = np.arctanh(A1)
    dZ1 = np.matmul(W2.T,dZ2)*(1-np.tanh(Z1)**2)

    dW1 = coeff*np.matmul(dZ1,X.T)
    dB1 = coeff*np.sum(dZ1, axis =1 , keepdims=1)
    dW2 = coeff*np.matmul(dZ2,A1.T)
    dB2 = coeff*np.sum(dZ2, axis =1, keepdims=1)

    return dW1, dB1, dW2, dB2

def train_neural_network(X, Y, num_hidden,num_iterations,learning_rate):
    """ Perform Gradient Descent to train the Neural Network

    Inputs:
        X: NumPy array (n, m) of feature data
        Y: NumPy array (m,) of labels
        num_hidden: int for number of hidden units in the hidden layer
        num_iterations: int for number of gradient descent iterations
        learning_rate: float for gradient descent learning rate

    Returns:
        W1: NumPy array for first layer trained weight parameters with shape (n_h, n_x)
        B1: NumPy array for first layer trained bias parameters with shape (n_h, 1)
        W2: NumPy array for second layer trained weight parameters with shape (1, n_h)
        B2: NumPy array for second layer trained bias parameters with shape (1, 1)
        costs: Python list of cost at each iteration
    """

    W1, B1, W2, B2 = initialize_parameters(X.ndim, num_hidden)
    costs = []

    for i in range(0,num_iterations):
        A1, A2 = forward_propagation(X,W1,B1,W2,B2)
        cost = compute_cost(A2,Y)
        costs.append(cost)
        dW1, dB1, dW2, dB2 = backward_propagation(X,Y,W2,A1,A2)
        W1 = W1 - learning_rate*dW1
        B1 = B1 - learning_rate*dB1
        W2 = W2 - learning_rate*dW2
        B2 = B2 - learning_rate*dB2

        if i % 1000 == 0:
            print(f"Iteration {i} - Cost: {cost}")

    return W1, B1, W2, B2, costs

def predict(X, W1, B1, W2, B2):
    """ Use the Neural Network to make predictions

    Inputs:
        X:  NumPy array of input samples of shape (n, m)
        W1: NumPy array of first layer of parameters with shape (n_h, n_x)
        B1: NumPy array of second layer bias parameters with shape (n_h, 1)
        W2: NumPy array of second layer of parameters with shape (1, n_h)
        B2: NumPy array of second layer bias parameters with shape (1, 1)

    Returns:
        NumPy array (m, ) of predictions.  Values are 0 or 1.
    """

    a1 = np.tanh(np.matmul(W1, X) + B1)
    a2 = sigmoid(np.matmul(W2, a1) + B2)
    predictions = np.rint(a2)


    return predictions

def compute_accuracy(X, Y, W1, B1, W2, B2):
    """ Compute the accuracy of the model

    Inputs:
        X:  NumPy array of feature data of shape (n, m)
        Y:  NumPy array of labels of shape (m,)
        W1: NumPy array of first layer of parameters with shape (n_h, n_x)
        B1: NumPy array of second layer bias parameters with shape (n_h, 1)
        W2: NumPy array of second layer of parameters with shape (1, n_h)
        B2: NumPy array of second layer bias parameters with shape (1, 1)

    Returns:
        NumPy array (m, ) of predictions.  Values are 0 or 1.
    """
    Y_predicted = predict(X, W1, B1, W2, B2)
    accuracy = np.mean(Y_predicted == Y)

    return accuracy

def plot_decision_boundary(X, Y, W1, B1, W2, B2, subplot=plt):
    """ Plot decision boundary

    Inputs:
        X:  NumPy array of training feature data of shape (n, m)
        Y:  NumPy array of training labels of shape (m,)
        W1: NumPy array of first layer of parameters with shape (n_h, n_x)
        B1: NumPy array of second layer bias parameters with shape (n_h, 1)
        W2: NumPy array of second layer of parameters with shape (1, n_h)
        B2: NumPy array of second layer bias parameters with shape (1, 1)
        subplot: Used when plotting subplots
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    # Generate a grid of points with distance h between them
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, W1, B1, W2, B2)
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    subplot.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    subplot.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
    #subplot.show()

# Uncomment one of the following to load the desired data set
#X, Y, n_iters, learning_rate = load_data_and_hyperparams('noisy_circles')
#X, Y, n_iters, learning_rate = load_data_and_hyperparams('noisy_moons')
#X, Y, n_iters, learning_rate = load_data_and_hyperparams('swirls')

plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.Spectral)
plt.title("")
plt.xlabel('feature x1')
plt.ylabel('feature x2')
plt.show()

print('Training for {0} iterations using a learning rate of {1}'.format(n_iters, learning_rate))

# Train
W1, B1, W2, B2, costs = train_neural_network(X, Y, 4, n_iters, learning_rate)

print(f"Accuracy: {compute_accuracy(X, Y, W1, B1, W2, B2)}")

# Plot cost
plt.plot(costs)
plt.title('Training Cost')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()

plot_decision_boundary(X, Y, W1, B1, W2, B2)




X, Y = load_partitioned_circles()

plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.Spectral)
plt.show()


# Creating a matplotlib with a 3x3 grid of subplots
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

for i in range(9):
    # Train the neural network with a specific number of hidden units
    print("Training with {0} hidden units".format(i+1))
    W1, B1, W2, B2, costs = train_neural_network(X, Y, i+1, 20000, 1)

    # Compute the training accuracy
    train_accuracy = compute_accuracy(X, Y, W1, B1, W2, B2)

    # Plot the decision boundary
    subplot = ax[int(i/3)][i%3]
    plot_decision_boundary(X, Y, W1, B1, W2, B2, subplot)

    subplot.set_xlim([-1.2, 1.2])
    subplot.set_ylim([-1.2, 1.2])
    subplot.title.set_text('n_h={0} acc:{1:0.3f} cost:{2:0.4f}'.format(
        i+1, train_accuracy, costs[-1]))
plt.show()


# Load noisy circles data set
X, Y, _, _ = load_data_and_hyperparams('noisy_circles')

# We rescale our input data to help speed up training. This is a technique
# called data preprocessing which we will discuss more of later
X= X* 1/np.max(X)

# Create a grid of subplots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

for i, n_h in enumerate([4, 10, 100]):

    # Train the neural network with a specific number of hidden units
    print("Training with {0} hidden units".format(n_h))
    W1, B1, W2, B2, costs = train_neural_network(X, Y, n_h, 25000, 1)

    # Plot the decision boundary
    subplot = ax[i%3]
    plot_decision_boundary(X, Y, W1, B1, W2, B2, subplot)

    subplot.set_xlim([-1, 1])
    subplot.set_ylim([-1, 1])
    subplot.title.set_text('n_h={0} acc:{1:0.3f} cost:{2:0.4f}'.format(
        n_h, compute_accuracy(X, Y, W1, B1, W2, B2), costs[-1]))
plt.show()
