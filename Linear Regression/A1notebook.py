import numpy as np
import math
import time
from a1_tools import load_defect_data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def non_vectorized_sigmoid(x):
    """ A non-vectorized implementation of the Sigmoid Function

    Inputs:
        x -- A single float or int

    Returns:
        A float for the Sigmoid function applied to x
    """
    s = None
    s = (1 / (1 + math.exp(-1 * x)))
    return s




def test_non_vectorized_sigmoid():
    """ Testcase for non_vectorized_sigmoid() """
    x = 1
    y_expected = 0.7310585
    y = non_vectorized_sigmoid(x)
    assert np.allclose(y, y_expected), 'Expected {0} but got {1} for an input of {2}'.format(y_expected, y, x)

    x = 0
    y_expected = 0.5
    y = non_vectorized_sigmoid(x)
    assert np.allclose(y, y_expected), 'Expected {0} but got {1} for an input of {2}'.format(y_expected, y, x)

    x = -1
    y_expected = 0.2689414
    y = non_vectorized_sigmoid(x)
    assert np.allclose(y, y_expected), 'Expected {0} but got {1} for an input of {2}'.format(y_expected, y, x)



    print('PASSED: test_non_vectorized_sigmoid()')


# Run the test
#est_non_vectorized_sigmoid()

"""This fails because the non non_vectorized_sigmoid cannot handle a vector
x = np.array([1,2,3])
print(non_vectorized_sigmoid(x))

"""
def sigmoid(x):
    """ A vectorized implementation of the Sigmoid Function

    Inputs:
        x -- A NumPy array, or a single float or int

    Returns:
        A NumPy array forthe Sigmoid function applied to x
    """
    s = (1 / (1 + np.exp(-1 * x)))


    return s

def test_sigmoid():
    """ Testcase for sigmoid() """

    x = np.array([1,2,3])
    y_expected = np.array([0.73105858, 0.88079708, 0.95257413])
    y = sigmoid(x)
    assert np.allclose(y, y_expected), 'Expected {0} but got {1} for an input of {2}'.format(y_expected, y, x)


    print('PASSED: test_sigmoid()')


# Run the test
#test_sigmoid()


"""
#compare vectorzied vs nonvectorized sigmoid
# Create a random input vector
x = np.random.rand(100000)

# The Non-vectorized Implementation which requires using
# a loop
t0 = time.time()
for elem in x:
    non_vectorized_sigmoid(elem)
t1 = time.time()

# The Non-vectorized Implementation which accepts the
# entire NumPy array
sigmoid(x)
t2 = time.time()

# Now print the runtime results
print('Non-vectorized runtime in seconds: {0}'.format(t1-t0))
print('Vectorized runtime in seconds:     {0}'.format(t2-t1))

"""

def initialize_parameters(n):
    """ Initialize the parameters with zeros for Logistic Regression

    Inputs:
        n: An int for number of input features

    Returns:
        NumPy Array: the W parameter vector of shape (n, 1)
        float: the b bias paramter
    """
    ### START CODE HERE ### (~ 2 line of code)
    # YOUR CODE HERE
    w = np.zeros((n,1))
    b = 0.0
    ### END CODE HERE ###

    assert(w.shape == (n, 1))
    assert(isinstance(b, float))

    return w, b

def test_initialize_parameters():
    """ Testcase for initialize_parameters() """

    n = 6
    w_expected = np.array([[0], [0], [0], [0], [0], [0]])
    b_expected = 0
    w, b = initialize_parameters(n)

    assert np.allclose(w, w_expected), 'For an input of {0}, expected w to be {1}, but got {2}'.format(n, w_expected, w)
    assert np.allclose(b, b_expected), 'For an input of {0}, expected b to be {1}, but got {2}'.format(n, b_expected, b)


    print('PASSED: test_initialize_parameters()')


# Run the test
#est_initialize_parameters()

"""
#testing vector formats in NumPy
# Creating some zero vectors of size 10
n = 10

col_vector = np.zeros((n, 1))
row_vector = np.zeros((1, n))
vector = np.zeros(n)

print('Shape of col_vector: {0}'.format(col_vector.shape))
print('Shape of row_vector: {0}'.format(row_vector.shape))
print('Shape of vector:     {0}'.format(vector.shape))
print()

print('Contents of col_vector:')
print(col_vector)
print()

print('Contents of row_vector:')
print(row_vector)
print()

print('Contents of vector - Note the number of brackets compared to row_vector:')
print(vector)

# Creating some vectors from a list
n = 10

col_vector = np.array([[1], [2], [3], [4]])
row_vector = np.array([[1, 2, 3, 4]])
vector = np.array([1, 2, 3, 4])

print('Shape of col_vector: {0}'.format(col_vector.shape))
print('Shape of row_vector: {0}'.format(row_vector.shape))
print('Shape of vector:     {0}'.format(vector.shape))
print()

print('Contents of col_vector:')
print(col_vector)
print()

print('Contents of row_vector:')
print(row_vector)
print()

print('Contents of vector - Note the number of brackets compared to row_vector:')
print(vector)

"""
#BroadCasting tests
	#Scalar and Array operations
"""
U = np.array([1, 2, 3, 4, 5, 6])
v = 10

Z = np.empty_like(U)  # Create an empty array with same shape as U
for i in range(U.shape[0]):
    Z[i] = U[i] + v

print('Result of adding v to each element of U:')
print(Z)


U = np.array([[1, 2, 3],
              [4, 5, 6]])
v = 10

# Replicate v into a new array VV
VV = np.tile(v, (U.shape))
Z = U + VV

print('VV:')
print(VV)
print()
print('Result of adding v to each element of U:')
print(Z)


U = np.array([[1, 2, 3],
              [4, 5, 6]])
v = 10

Z = U + v

print('Result of Broadcasting on U+v:')
print(Z)

"""
	#Operating on Arrays of Different Size

U = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
V = np.array([3, 4, 5])

# Replicate v into a new array VV
"""
VV = np.tile(V, (U.shape[0], 1))
Z = U + VV

print('VV:')
print(VV)
print()
print('Result of adding V to each row of U:')
print(Z)

U = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
V = np.array([3, 4, 5])

Z = U + V

print('Result of Broadcasting and U+V:')
print(Z)

U = np.array([[1, 2, 3]])
V = np.array([[4], [5], [6], [7]])
print(U+V)
"""

#Implementing hte Hypothesis Function

#A = sigmoid(w^t X+b)
#𝑋 is the (n, m) input matrix where 𝑛 is the number of features, and 𝑚 is the number of data samples.
#𝑤 is the (n, 1) parameter matrix.
#𝑏 is the bias parameter.
#𝐴 is a (1, m) vector which represents the hypothesis for each of the 𝑚 samples.

def hypothesis(X, w, b):
    """
    Inputs:
        X: NumPy array of input samples of shape (n, m)
        w: NumPy array of parameters with shape (n, 1)
        b: float for the bias parameter

    Returns:
        NumPy array of shape (1, m) with the hypothesis of each sample
    """
    A = None


    A = np.matmul(w.T, X) + b
    A = sigmoid(A)


    return A

def test_hypothesis():
    """ Testcase for hypothesis() """

    X = np.array([[1, 0.5, 0.23],
                  [0.95, 0.43, 0.14],
                  [0.78, 0.33, 0.31]])
    w = np.array([[1.55], [0.25], [0.1]])
    b = 0.13

    A_expected = np.array([[0.8803238, 0.73990984, 0.63471542]])
    A = hypothesis(X, w, b)

    assert isinstance(A, np.ndarray), 'Expected a Numpy array for A but got {0}'.format(type(A))
    assert A.shape == A_expected.shape, 'Unexpected shape for A. Expected {0} but got {1}'.format(
        A_expected.shape, A.shape)
    assert np.allclose(A, A_expected), 'expected A to be {0}, but got {1}'.format(A_expected, A)



    print('PASSED: test_hypothesis()')


# Run the test
#test_hypothesis()

#Computing the Cost

def compute_cost(A, Y):
    """ Vectorized Logistic Regression Cost Function

    Inputs:
        A: NumPy array of shape (1, m)
        Y: NumPy array (m, ) of known labels

    Returns:
        A single float for the cost
    """
    #𝐽=(-1/m)*(∑ y^(𝑖)*log(𝑎^(𝑖)) + ∑(1−𝑦^(𝑖))*log(1−𝑎^(𝑖)))
    cost = None

    sum_one = np.sum(Y * np.log(A))
    sum_two = np.sum((1 - Y) * np.log(1 - A))
    cost = -1 * (1 / len(Y)) * (sum_one + sum_two)

   	#OR
   	# cost = (-1/len(Y))* (np.sum(Y * np.log(A))+np.sum((1 - Y) * np.log(1 - A)))

    return cost
def test_compute_cost():
    """ Testcase for compute_cost() """

    A = np.array([[0.1, 0.5, 0.75]])
    Y = np.array([1, 0, 1])

    cost_expected = 1.09447145
    cost = compute_cost(A, Y)

    assert isinstance(cost, float), 'Expected a float for cost but got {0}'.format(type(cost))
    assert np.allclose(cost, cost_expected), 'expected cost to be {0}, but got {1}'.format(cost_expected, cost)


    print('PASSED: test_compute_cost()')


# Run the test
#test_compute_cost()

#Computing Gradients

def compute_gradients(A, X, Y):
    """ Compute the gradients of the cost function

    Inputs:
        A: NumPy array of shape (1, m)
        X: NumPy array of shape (n, m)
        Y: NumPy array of shape (m, )

    Returns:
        Two NumPy arrays. One for the cost derivative w.r.t. dw
        and one for the cost derivative w.r.t. db
    """
    dw = None
    db = None

    m = A.shape[1]
    dw = (1 / m) * np.matmul(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == (X.shape[0], 1))

    return dw, db

def test_compute_gradients():
    """ Testcase for compute_gradients() """

    A = np.array([[0.99, 0.21, 0.87]])
    X = np.array([[1, 0.5, 0.23],
                  [0.95, 0.43, 0.14],
                  [0.78, 0.33, 0.31]])
    Y = np.array([[1, 0, 1]])

    dw_expected = np.array([[0.0217    ],
                            [0.02086667],
                            [0.00706667]])
    db_expected = 0.0233333333
    dw, db = compute_gradients(A, X, Y)

    assert isinstance(dw, np.ndarray), 'Expected a Numpy array for dw but got {0}'.format(type(dw))
    assert dw.shape == dw_expected.shape, 'Unexpected shape for dw. Expected {0} but got {1}'.format(
        dw_expected.shape, dw.shape)
    assert np.allclose(dw, dw_expected), 'expected dw to be {0}, but got {1}'.format(dw_expected, dw)

    assert isinstance(db, float), 'Expected a float for db but got {0}'.format(type(db))
    assert np.allclose(db, db_expected), 'expected db to be {0}, but got {1}'.format(db_expected, db)



    print('PASSED: test_compute_gradients()')


# Run the test
#test_compute_gradients()

#Gradient Descent
def gradient_descent(X, Y, num_iterations, learning_rate, print_costs=True):
    """ Perform Gradient Descent for Logistic Regression

    Inputs:
        X: NumPy array (n, m)
        Y: NumPy array (m,)
        num_iterations: int for number of gradient descent iterations
        learning_rate: float for gradient descent learning rate
        print_costs: bool to enable printing of costs

    Returns:
        w: NumPy array for trained parameters w
        b: float for trained bias parameter b
        costs: Python list of cost at each iteration
    """
    w, b = None, None

    ### START CODE HERE ### (~1  line of code)
    # YOUR CODE HERE
    # initialize_parameters(...) returns w, b
    w, b = initialize_parameters(X.shape[0])
    ### END CODE HERE ###

    # We will use a list to store the cost at each iteration
    # so that we can plot this later for educational purposes
    costs = []

    for i in range(num_iterations):

        ### START CODE HERE ### (~5  line of code)
        # YOUR CODE HERE
        A = hypothesis(X, w, b) # perform regression
        dw, db = compute_gradients(A, X, Y)
        w = w - (learning_rate * dw) # adjust weight parameter
        b = b - (learning_rate * db) # adjust bias parameter
        cost = compute_cost(A, Y) # compute cost of predicted A
        ### END CODE HERE ###

        # Convert and save the cost at this iteration
        costs.append(cost)

        # Print cost after ever 5000 iterations
        if print_costs and i % 5000 == 0:
            print("Iteration {0} - Cost: {1}".format(i, str(costs[-1])))

    return w, b, costs
def test_gradient_descent_one_iteration():
    """ Testcase for gradient_descent for one iteration() """

    X = np.array([[71.99, 57.95, 73.10, 63.45, 82.74, 18.05, 80.31, 3.76, 66.02, 42.84],
                   [67.21, 1.41, 30.45, 10.01, 97.86, 6.09, 52.75, 19.03, 80.13, 48.92]])
    Y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    num_iterations = 1
    learning_rate = 0.001182

    w_expected = np.array([[-0.01796699],
                           [-0.01452442]])
    b_expected = -0.0001182
    costs_expected = [0.69314718,]

    w, b, costs = gradient_descent(X, Y, num_iterations, learning_rate, print_costs=False)

    assert isinstance(w, np.ndarray), 'Expected a Numpy array for w but got {0}'.format(type(w))
    assert w.shape == w_expected.shape, 'Unexpected shape for w. Expected {0} but got {1}'.format(
        w_expected.shape, w.shape)
    assert np.allclose(w, w_expected), 'expected w to be {0}, but got {1}'.format(w_expected, w)

    assert isinstance(b, float), 'Expected a float for b but got {0}'.format(type(b))
    assert np.allclose(b, b_expected), 'expected b to be {0}, but got {1}'.format(b_expected, b)

    assert isinstance(costs, list), 'Expected a Python list for costs but got {0}'.format(type(costs))
    assert len(costs) == len(costs_expected), 'Unexpected length for costs. Expected {0} but got {1}'.format(
        len(costs_expected), len(costs))
    assert np.allclose(costs, costs_expected), 'expected costs to be {0}, but got {1}'.format(costs_expected, costs)


    print('PASSED: test_gradient_descent_one_iteration()')


# Run the test
#test_gradient_descent_one_iteration()

def test_gradient_descent_multiple_iterations():
    """ Testcase for gradient_descent() for multiple iterations """

    X = np.array([[71.99, 57.95, 73.10, 63.45, 82.74, 18.05, 80.31, 3.76, 66.02, 42.84],
                   [67.21, 1.41, 30.45, 10.01, 97.86, 6.09, 52.75, 19.03, 80.13, 48.92]])
    Y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    num_iterations = 3
    learning_rate = 0.001182

    w_expected = np.array([[-0.01149753],
                           [-0.00984429]])
    b_expected = 0.00029081894

    costs_expected = [0.6931471, 0.5818799, 0.5573549]

    w, b, costs = gradient_descent(X, Y, num_iterations, learning_rate, print_costs=False)

    assert isinstance(w, np.ndarray), 'Expected a Numpy array for w but got {0}'.format(type(w))
    assert w.shape == w_expected.shape, 'Unexpected shape for w. Expected {0} but got {1}'.format(
        w_expected.shape, w.shape)
    assert np.allclose(w, w_expected), 'expected w to be {0}, but got {1}'.format(w_expected, w)

    assert isinstance(b, float), 'Expected a float for b but got {0}'.format(type(b))
    assert np.allclose(b, b_expected), 'expected b to be {0}, but got {1}'.format(b_expected, b)

    assert isinstance(costs, list), 'Expected a Python list for costs but got {0}'.format(type(costs))
    assert len(costs) == len(costs_expected), 'Unexpected length for costs. Expected {0} but got {1}'.format(
        len(costs_expected), len(costs))
    assert np.allclose(costs, costs_expected), 'expected costs to be {0}, but got {1}'.format(costs_expected, costs)


    print('PASSED: test_gradient_descent_multiple_iterations()')


# Run the test
#test_gradient_descent_multiple_iterations()
def predict(X, w, b):
    """ Use the Logistic Regression Model to make predictions

    Inputs:
        X: NumPy array (n, m) of feature data
        w: NumPy array (n, 1) of trained parameters w
        b: float for trained bias parameter

    Returns:
        NumPy array (m, ) of predictions.  Values are 0 or 1.
    """
    y_pred = None

    y_pred = np.rint(sigmoid((np.matmul(w.T, X) + b)))

    return y_pred

#X, Y = load_defect_data()

print(X.shape)
print(Y.shape)
print('Range of feature 0: {0} to {1}'.format(np.min(X[0,:]), np.max(X[0,:])))
print('Range of feature 1: {0} to {1}'.format(np.min(X[1,:]), np.max(X[1,:])))
for i in range(10):
    print('Sample {0}: x: {1} y: {2}'.format(i, X[:,i], Y[i]))



#%matplotlib inline

#cmap = plt.cm.Spectral
#plt.scatter(X[0,:], X[1,:], c=Y, cmap=cmap)
#plt.xlabel('test 1')
#plt.ylabel('test 2')

#plt.legend(handles=[mpatches.Patch(color=cmap(0), label='Not Defective'),mpatches.Patch(color=cmap(cmap.N), label='Defective')])

#plt.title("Chip Defects")
#plt.show()


w, b, costs = gradient_descent(X, Y, 80000, 0.001182)

# Expected final cost: 0.22623328858976322
print('Final cost: {0}'.format(costs[-1]))
print()

# Expected output:
# [[-0.05769606]
#  [-0.06118479]]
print(w)
print()

# Expected output: 5.410521971317577
print(b)

#plt.plot(costs)
#plt.xlabel('Iteration')
#plt.ylabel('Cost')
#plt.title("Logistic Regression Training Cost Progression")
#plt.show()

def test_predict():
    """ Testcase for predict() """

    X = np.array([[59.30, 61.80, 77.68,  9.35],
                  [72.73, 68.96, 17.92, 35.19]])

    w = np.array([[-0.05769606],
                  [-0.06118479]])
    b = 5.41052197

    Y_expected = np.array([[0, 0, 0, 1]])
    Y = predict(X, w, b)

    assert isinstance(Y, np.ndarray), 'Expected a Numpy array for Y but got {0}'.format(type(Y))
    assert Y.shape == Y_expected.shape, 'Unexpected shape for Y. Expected {0} but got {1}'.format(
        Y_expected.shape, Y.shape)
    assert np.allclose(Y, Y_expected), 'expected Y to be {0}, but got {1}'.format(Y_expected, Y)



    print('PASSED: test_predict()')


# Run the test
#test_predict()

def plot_decision_boundary(X, Y, w, b):
    """ Plot decision boundary for 2D data

    Inputs:
        X: NumPy array (2, m) of feature data
        Y: NumPy array (m,)   of label data
        w: NumPy array (n, 1) of trained parameters w
        b: float for the bias parameter
    """
    # Plot the original data
    plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.Spectral)

    # Plot the decision boundary line
    # Pick the extremes of x_0 (and go beyond a little bit)
    # Compute x_1 for these values according to the decision
    # boundary equation, and plot the line
    x_0 = np.array([min(X[0,:]) - 1, max(X[0,:]) + 1])
    x_1 = - (w[0] * x_0 + b) / w[1]
    plt.plot(x_0, x_1, label = "Decision_Boundary")

    # scale the plot and other formatting
    axes = plt.gca()
    axes.set_xlim([min(X[0,:]),max(X[0,:])])
    axes.set_ylim([min(X[1,:]),max(X[1,:])])
    plt.ylabel('test 1')
    plt.xlabel('test 2')
    plt.show()

#plot_decision_boundary(X, Y, w, b)

# GRADED FUNCTION: compute_accuracy

def compute_accuracy(X, Y, w, b):
    """ Compute the accuracy of a trained Logistic Regression
        model described by its trained parameters

    Inputs:
        X: NumPy array (n, m) feature data
        Y: NumPy array (m, ) known labels for feature data X
        w: NumPy array (n, 1) trained model parameters w
        b: float for trained bias parameter

    Returns:
        float between 0 and 1 denoting the accuracy of the
        Logistic Regression model
    """
    accuracy = None

    ### START CODE HERE ### (1-2  line of code)
    # YOUR CODE HERE
    correct = np.sum(predict(X, w, b) == Y)
    accuracy = correct / max(Y.shape[0], Y.T.shape[0])
    ### END CODE HERE ###
    return accuracy

def test_compute_accuracy():
    """ Testcase for compute_accuracy() """

    X = np.array([[59.30, 61.80, 77.68,  9.35],
                  [72.73, 68.96, 17.92, 35.19]])

    w = np.array([[-0.05769606],
                  [-0.06118479]])
    b = 5.41052197
    Y = np.array([[0, 1, 0, 1]])

    accuracy_expected = 0.75

    accuracy = compute_accuracy(X, Y, w, b)

    assert isinstance(accuracy, float), 'Expected a float for accuracy but got {0}'.format(type(accuracy))
    assert np.allclose(accuracy, accuracy_expected), 'expected db to be {0}, but got {1}'.format(accuracy_expected, accuracy)


    print('PASSED: test_compute_accuracy()')


# Run the test
test_compute_accuracy()
accuracy = compute_accuracy(X, Y, w, b)
print('Accuracy: {0}'.format(accuracy))

def train_and_assess(X, Y, num_iterations=80000, learning_rate=0.001182):
    """ Trains a Logistic Regression model, then access its accuracy
        and decision boundary on the training data.

    Inputs:
        X: NumPy array (n, m) feature data
        Y: NumPy array (m, ) known labels for feature data X
        num_iterations: int for number of gradient descent iterations
        learning_rate: float for gradient descent learning rate
    """

    # Train model
    w, b, costs = gradient_descent(X, Y, num_iterations, learning_rate)
    print('Final Cost: {0}'.format(costs[-1]))

    # Plot Cost during training
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title("Logistic Regression Training Cost Progression")
    plt.show()

    # Plot decision boundary
    plot_decision_boundary(X, Y, w, b)

    # Print accuracy
    accuracy = compute_accuracy(X, Y, w, b)
    print('Accuracy: {0}'.format(accuracy))

train_and_assess(X, Y, num_iterations=1000)
train_and_assess(X, Y, num_iterations=10000)
train_and_assess(X, Y, num_iterations=300000)
train_and_assess(X, Y, learning_rate=0.0001)
train_and_assess(X, Y, num_iterations=100, learning_rate=0.002)

#from a1_tools import load_swirls, load_noisy_circles, load_noisy_moons, load_partitioned_circles

# Uncomment only one of these lines to load the desired data set
#X, Y = load_swirls()
# X, Y = load_noisy_circles()
# X, Y = load_noisy_moons()
# X, Y = load_partitioned_circles()


# Plot the data
#plt.scatter(X[0,:], X[1,:], c=Y, cmap=plt.cm.Spectral)
#plt.show()

# Now train and assess.  NOTE: The default learning rate and number of
# training iterations were tuned for our linearly separable data set
# you may want to play around with those numbers.
train_and_assess(X, Y, num_iterations=10000, learning_rate = 0.2)
