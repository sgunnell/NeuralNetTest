
import numpy as np
import matplotlib.pyplot as plt
from a2_tools import sigmoid
from  neuralnetworks import *

#init all trainable parameters randomly

def test_initialize_parameters():
    """ Testcase for initialize_parameters() """

    # Set up test inputs
    n_x = 2
    n_h = 4
    w1_expected = np.array([[ 0.01764052,  0.00400157],
                            [ 0.00978738,  0.02240893],
                            [ 0.01867558, -0.00977278],
                            [ 0.00950088, -0.00151357]])
    b1_expected = np.array([[0.],
                            [0.],
                            [0.],
                            [0.]])

    # Set up expected outputs
    w2_expected = np.array([[-0.00103219,  0.00410599,  0.00144044,  0.01454274]])
    b2_expected = np.array([[0.]])

    # Exercise function under test
    w1, b1, w2, b2 = initialize_parameters(n_x, n_h)

    # Perform test assertions
    assert isinstance(w1, np.ndarray), 'Expected a Numpy array for w1 but got {0}'.format(type(w1))
    assert w1.shape == w1_expected.shape, 'Unexpected shape for w1. Expected {0} but got {1}'.format(
        w1_expected.shape, w1.shape)
    assert np.allclose(w1, w1_expected), 'expected w1 to be {0}, but got {1}'.format(w1_expected, w1)

    assert isinstance(b1, np.ndarray), 'Expected a Numpy array for b1 but got {0}'.format(type(b1))
    assert b1.shape == b1_expected.shape, 'Unexpected shape for b1. Expected {0} but got {1}'.format(
        b1_expected.shape, b1.shape)
    assert np.allclose(b1, b1_expected), 'expected w1 to be {0}, but got {1}'.format(b1_expected, b1)

    assert isinstance(w2, np.ndarray), 'Expected a Numpy array for w2 but got {0}'.format(type(w2))
    assert w2.shape == w2_expected.shape, 'Unexpected shape for w2. Expected {0} but got {1}'.format(
        w2_expected.shape, w2.shape)
    assert np.allclose(w2, w2_expected), 'expected w2 to be {0}, but got {1}'.format(w2_expected, w2)

    assert isinstance(b2, np.ndarray), 'Expected a Numpy array for b2 but got {0}'.format(type(b2))
    assert b2.shape == b2_expected.shape, 'Unexpected shape for b2. Expected {0} but got {1}'.format(
        b2_expected.shape, b2.shape)
    assert np.allclose(b2, b2_expected), 'expected w2 to be {0}, but got {1}'.format(b2_expected, b2)



    print('PASSED: test_initialize_parameters()')

def test_forward_propagation():
    """ Testcase for forward_propagation() """

    # Set up test inputs
    X = np.array([[1.25,-0.28, 0.30],
                  [0.37, 0.87, 0.96]])
    W1 = np.array([[ 0.21, -0.84],
                   [-0.10, 0.26],
                   [ 0.29, -1.11 ],
                   [ 0.29,-1.14 ]])
    B1 = np.array([[-0.09], [ 0.08], [ 0.34], [ 0.38]])
    W2 = np.array([[ 1.22, -0.30, 1.86, 1.90]])
    B2 = np.array([[-0.39772327]])

    # Set up expected outputs
    A1_expected = np.array(
        [[-0.13742494, -0.70621888, -0.68229742],
         [ 0.05115531,  0.3222896,   0.29094652],
         [ 0.28379069, -0.60872922, -0.56394556],
         [ 0.31013973, -0.5999058,  -0.55625932]])
    A2_expected = np.array([[0.63097175, 0.02588024, 0.03157695]])

    # Exercise function under test
    A1, A2 = forward_propagation(X, W1, B1, W2, B2)

    # Perform test assertions
    assert isinstance(A1, np.ndarray), 'Expected a Numpy array for A1 but got {0}'.format(type(A1))
    assert A1.shape == A1_expected.shape, 'Unexpected shape for A1. Expected {0} but got {1}'.format(
        A1_expected.shape, A1.shape)
    assert np.allclose(A1, A1_expected), 'expected A1 to be {0}, but got {1}'.format(A1_expected, A1)

    assert isinstance(A2, np.ndarray), 'Expected a Numpy array for A2 but got {0}'.format(type(A2))
    assert A2.shape == A2_expected.shape, 'Unexpected shape for A2. Expected {0} but got {1}'.format(
        A2_expected.shape, A2.shape)
    assert np.allclose(A2, A2_expected), 'expected A2 to be {0}, but got {1}'.format(A2_expected, A2)


    print('PASSED: test_forward_propagation()')

def test_compute_cost():
    """ Testcase for compute_cost() """

    # Set up test inputs
    A2 = np.array([[0.63, 0.03, 0.03, 0.88]])
    Y  = np.array([0, 0, 0, 1])

    # Set up expected outputs
    cost_expected = 0.295751014

    # Exercise function under test
    cost = compute_cost(A2, Y)

    # Perform test assertions
    assert isinstance(cost, float), 'Expected a float for cost but got {0}'.format(type(cost))
    assert np.allclose(cost, cost_expected), 'expected cost to be {0}, but got {1}'.format(cost_expected, cost)


    print('PASSED: test_compute_cost()')

def test_backward_propagation():
    """ Testcase for backward_propagation() """

    # Set up test inputs
    X = np.array([[1.25,-0.28, 0.30, 0.50],
                  [0.37, 0.87, 0.96,-0.13]])
    Y  = np.array([0, 0, 0, 1])
    W2 = np.array([[ 1.22, -0.30, 1.86, 1.90]])
    A1 = np.array([[-0.14,-0.71,-0.68, 0.13],
                   [ 0.05, 0.32, 0.29,-0.01],
                   [ 0.28,-0.61,-0.56, 0.56],
                   [ 0.31,-0.60,-0.56, 0.59]])
    A2 = np.array([[0.63, 0.03, 0.03, 0.88]])

    # Set up expected outputs
    dW1_expected = np.array([[ 0.21769431,  0.08304951],
                             [-0.05446804, -0.02234407],
                             [ 0.31874785,  0.12168553],
                             [ 0.3199167 ,  0.122237  ]])
    dB1_expected = np.array([[ 0.16185892],
                             [-0.04221315],
                             [ 0.25001609],
                             [ 0.25223498]])
    dW2_expected = np.array([[-0.036375, 0.01275, 0.018525, 0.022425]])
    dB2_expected = np.array([[0.1425]])

    # Exercise function under test
    dW1, dB1, dW2, dB2 = backward_propagation(X, Y, W2, A1, A2)

    # Perform test assertions
    assert isinstance(dW1, np.ndarray), 'Expected a Numpy array for dW1 but got {0}'.format(type(dW1))
    assert dW1.shape == dW1_expected.shape, 'Unexpected shape for dW1. Expected {0} but got {1}'.format(
        dW1_expected.shape, dW1.shape)
    assert np.allclose(dW1, dW1_expected), 'expected dW1 to be {0}, but got {1}'.format(dW1_expected, dW1)

    assert isinstance(dB1, np.ndarray), 'Expected a Numpy array for dB1 but got {0}'.format(type(dB1))
    assert dB1.shape == dB1_expected.shape, 'Unexpected shape for dB1. Expected {0} but got {1}'.format(
        dB1_expected.shape, dB1.shape)
    assert np.allclose(dB1, dB1_expected), 'expected dB1 to be {0}, but got {1}'.format(dB1_expected, dB1)

    assert isinstance(dW2, np.ndarray), 'Expected a Numpy array for dW2 but got {0}'.format(type(dW2))
    assert dW2.shape == dW2_expected.shape, 'Unexpected shape for dW2. Expected {0} but got {1}'.format(
        dW2_expected.shape, dW2.shape)
    assert np.allclose(dW2, dW2_expected), 'expected dW2 to be {0}, but got {1}'.format(dW2_expected, dW2)

    assert isinstance(dB2, np.ndarray), 'Expected a Numpy array for dB2 but got {0}'.format(type(dB2))
    assert dB2.shape == dB2_expected.shape, 'Unexpected shape for dB2. Expected {0} but got {1}'.format(
        dB2_expected.shape, dB2.shape)
    assert np.allclose(dB2, dB2_expected), 'expected dB2 to be {0}, but got {1}'.format(dB2_expected, dB2)


    print('PASSED: test_backward_propagation()')


def test_train_neural_network_one_iteration():
    """ Testcase for train_neural_network() for one iteration """

    # Set up test inputs
    X = np.array([[1.25,-0.28, 0.30, 0.50],
                  [0.37, 0.87, 0.96,-0.13]])
    Y  = np.array([0, 0, 0, 1])
    num_hidden = 4
    num_iterations = 1
    learning_rate = 1.8

    # Set up expected outputs
    W1_expected = np.array([[ 0.01781924,  0.00454265],
                            [ 0.00907641,  0.02025719],
                            [ 0.01842608, -0.01052785],
                            [ 0.00698118, -0.00913767]])
    B1_expected = np.array([[ 0.0004644 ],
                            [-0.0018467 ],
                            [-0.00064811],
                            [-0.00654455]])
    W2_expected = np.array([[-0.0061861,  -0.00933635,  0.00332793,  0.0136899 ]])
    B2_expected = np.array([[-0.4500493]])
    costs_expected = [0.69318027,]

    # Exercise function under test
    W1, B1, W2, B2, costs = train_neural_network(X, Y, num_hidden, num_iterations, learning_rate)

    # Perform test assertions
    assert isinstance(W1, np.ndarray), 'Expected a Numpy array for W1 but got {0}'.format(type(W1))
    assert W1.shape == W1_expected.shape, 'Unexpected shape for W1. Expected {0} but got {1}'.format(
        W1_expected.shape, W1.shape)
    assert np.allclose(W1, W1_expected), 'expected W1 to be {0}, but got {1}'.format(W1_expected, W1)

    assert isinstance(B1, np.ndarray), 'Expected a Numpy array for B1 but got {0}'.format(type(B1))
    assert B1.shape == B1_expected.shape, 'Unexpected shape for B1. Expected {0} but got {1}'.format(
        B1_expected.shape, B1.shape)
    assert np.allclose(B1, B1_expected), 'expected B1 to be {0}, but got {1}'.format(B1_expected, B1)

    assert isinstance(W2, np.ndarray), 'Expected a Numpy array for W2 but got {0}'.format(type(W2))
    assert W2.shape == W2_expected.shape, 'Unexpected shape for W2. Expected {0} but got {1}'.format(
        W2_expected.shape, W2.shape)
    assert np.allclose(W2, W2_expected), 'expected W2 to be {0}, but got {1}'.format(W2_expected, W2)

    assert isinstance(B2, np.ndarray), 'Expected a Numpy array for B2 but got {0}'.format(type(B2))
    assert B2.shape == B2_expected.shape, 'Unexpected shape for B2. Expected {0} but got {1}'.format(
        B2_expected.shape, B2.shape)
    assert np.allclose(B2, B2_expected), 'expected B2 to be {0}, but got {1}'.format(B2_expected, B2)

    assert isinstance(costs, list), 'Expected a Python list for costs but got {0}'.format(type(costs))
    assert len(costs) == len(costs_expected), 'Unexpected length for costs. Expected {0} but got {1}'.format(
        len(costs_expected), len(costs))
    assert np.allclose(costs, costs_expected), 'expected costs to be {0}, but got {1}'.format(costs_expected, costs)


    print('PASSED: test_train_neural_network_one_iteration()')


def test_train_neural_network_multiple_iterations():
    """ Testcase for train_neural_network() for more than one iteration """

    # Set up test inputs
    X = np.array([[1.25,-0.28, 0.30, 0.50],
                  [0.37, 0.87, 0.96,-0.13]])
    Y  = np.array([0, 0, 0, 1])
    num_hidden = 4
    num_iterations = 5
    learning_rate = 1.8

    # Set up expected outputs
    W1_expected = np.array([[ 0.01888309,  0.02084325],
                            [ 0.01090226,  0.05239335],
                            [ 0.01777069, -0.02252877],
                            [ 0.00478214, -0.04101898]])
    B1_expected = np.array([[ 0.00561   ],
                            [ 0.00772194],
                            [-0.00416679],
                            [-0.01681616]])
    W2_expected = np.array([[-0.02265813, -0.05149088, 0.02090487, 0.04420093]])
    B2_expected = np.array([[-0.99265906]])
    costs_expected = [0.69318027, 0.60564122, 0.577927463, 0.56815093, 0.56427790]

    # Exercise function under test
    W1, B1, W2, B2, costs = train_neural_network(X, Y, num_hidden, num_iterations, learning_rate)

    # Perform test assertions
    assert isinstance(W1, np.ndarray), 'Expected a Numpy array for W1 but got {0}'.format(type(W1))
    assert W1.shape == W1_expected.shape, 'Unexpected shape for W1. Expected {0} but got {1}'.format(
        W1_expected.shape, W1.shape)
    assert np.allclose(W1, W1_expected), 'expected W1 to be {0}, but got {1}'.format(W1_expected, W1)

    assert isinstance(B1, np.ndarray), 'Expected a Numpy array for B1 but got {0}'.format(type(B1))
    assert B1.shape == B1_expected.shape, 'Unexpected shape for B1. Expected {0} but got {1}'.format(
        B1_expected.shape, B1.shape)
    assert np.allclose(B1, B1_expected), 'expected B1 to be {0}, but got {1}'.format(B1_expected, B1)

    assert isinstance(W2, np.ndarray), 'Expected a Numpy array for W2 but got {0}'.format(type(W2))
    assert W2.shape == W2_expected.shape, 'Unexpected shape for W2. Expected {0} but got {1}'.format(
        W2_expected.shape, W2.shape)
    assert np.allclose(W2, W2_expected), 'expected W2 to be {0}, but got {1}'.format(W2_expected, W2)

    assert isinstance(B2, np.ndarray), 'Expected a Numpy array for B2 but got {0}'.format(type(B2))
    assert B2.shape == B2_expected.shape, 'Unexpected shape for B2. Expected {0} but got {1}'.format(
        B2_expected.shape, B2.shape)
    assert np.allclose(B2, B2_expected), 'expected B2 to be {0}, but got {1}'.format(B2_expected, B2)

    assert isinstance(costs, list), 'Expected a Python list for costs but got {0}'.format(type(costs))
    assert len(costs) == len(costs_expected), 'Unexpected length for costs. Expected {0} but got {1}'.format(
        len(costs_expected), len(costs))
    assert np.allclose(costs, costs_expected), 'expected costs to be {0}, but got {1}'.format(costs_expected, costs)


    print('PASSED: test_train_neural_network_multiple_iterations()')


def test_predict():
    """ Testcase for predict() """

    # Set up test inputs
    X = np.array([[1.25,-0.28, 0.30, 0.50],
                  [0.37, 0.87, 0.96,-0.13]])
    W1 = np.array([[ 0.21, -0.84],
                   [-0.10, 0.26],
                   [ 0.29, -1.11 ],
                   [ 0.29, -1.14 ]])
    B1 = np.array([[-0.09], [ 0.08], [ 0.34], [ 0.38]])
    W2 = np.array([[ 1.22, -0.30, 1.86, 1.90]])
    B2 = np.array([[-0.39772327]])

    # Set up expected outputs
    predictions_expected = np.array([[1., 0., 0., 1.]])

    # Exercise function under test
    predictions = predict(X, W1, B1, W2, B2)

    # Perform test assertions
    assert isinstance(predictions, np.ndarray), 'Expected a Numpy array for predictions but got {0}'.format(type(predictions))
    assert predictions.shape == predictions_expected.shape, 'Unexpected shape for predictions. Expected {0} but got {1}'.format(
        predictions_expected.shape, predictions.shape)
    assert np.allclose(predictions, predictions_expected), 'expected predictions to be {0}, but got {1}'.format(predictions_expected, predictions)


    print('PASSED: test_predict()')


# Run the test
test_initialize_parameters()
test_forward_propagation()
test_compute_cost()
test_backward_propagation()
test_train_neural_network_one_iteration()
test_train_neural_network_multiple_iterations()
test_predict()
