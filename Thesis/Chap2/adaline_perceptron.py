'''
PERCEPTRON AND ADALINE CODE 

X:      training set. Can be more than two features.
sigma:  output. Same len as X.
W:      initial weights. len_W = nb_features+1. Can be set to zero or randomly initialized.

theta:  threshold.      0 in perceptron, other in adaline
lr:     learning rate.  Historically 1 in percetron, close to zero in adaline.
epochs: number of iterations over the training set. 

'''

# Adaline or Perceptron
ADALINE = 0 # 0 for perceptron, 1 for adaline

# Training set
X = [[1,5],[2,3],[7,1],[4,1],[5,4]]
sigma = [1,0,1,0,1]
W = [-9,1,3] # W = np.random.randn(len(X[0])+1)

# Training parameters.
theta=0.1   # 0 in perceptron
lr=0.3      # 1 in perceptron
epochs=10



import numpy as np
import matplotlib.pyplot as plt

def sum_func(W,x_i):
    '''
    W: weights
    x_i: inputs
    '''
    return np.dot(W.T,x_i)

def act_func(sum_res, theta=0):
    '''
    sum_res: result of summation function
    theta: threshold
    '''
    if ADALINE: 
        return 1.0 if (sum_res > theta) else 0.0
    else:
        return 1.0 if (sum_res > 0) else 0.0

def err_func(sigma_act, sigma_pred):
    '''
    sigma_act: actual output
    sigma_pred: predicted output
    '''
    if ADALINE: 
        return (sigma_act - sigma_pred)
    else:
        return sigma_act

def update_weights(W, lr, err, x_i):
    if ADALINE: 
        return W + lr * err * x_i
    else:
        return W + err * x_i

def train_ANN(X, sigma, W=None, epochs=10, lr=0.3, theta=0):
    '''
    X: inputs with m training examples, n features
    sigma: outputs
    W: weights
    epochs: number of iterations
    lr: learning rate.
    theta: threshold
    '''
    X = np.array(X)
    m, n = X.shape
    W = np.zeros(n+1) if W==None else np.array(W) # could also be W = np.random.randn(len(X[0])+1)
    n_miss_list = [] # How many examples were misclassified at every iteration
    # Training.
    for epoch in range(epochs):
        n_miss = 0 # Storing amount of misclassified
        for idx, x_i in enumerate(X):
            x_i = np.append(1, x_i) # x0 = 1
            sum_res = sum_func(W, x_i)
            sigma_pred = act_func(sum_res, theta)
            if sigma_pred != sigma[idx]:
                print(f'error: {sigma[idx]}, {sigma_pred}')
                err = err_func(sigma[idx], sigma_pred)
                W = update_weights(W, x_i, err, lr)
                n_miss += 1
            print(W)
        print(f'{n_miss} miscassified at epoch {epoch}\n')
        n_miss_list.append(n_miss)
    return W, n_miss_list

def plot_results_ANN(X, sigma, W):
    '''
    X:      inputs
    sigma:  outputs
    W:      weights
    '''
    # Line w0 + w1 x + w2 y = 0 -> y = -w0/w2 - w1/w2 x
    X=np.array(X)
    x1 = np.array( [0, -W[0]/W[1]] )
    x2 = np.array( [-W[0]/W[2], 0] )
    
    # Plotting
    fig = plt.figure(figsize=(10,8))
    for idx, x_i in enumerate(X):
        if sigma[idx] == 1:
            plt.plot(x_i[0], x_i[1], 'ro')
        else:
            plt.plot(x_i[0], x_i[1], 'b.')
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Adaline Algorithm')
    plt.plot(x1, x2, 'y-')
    fig.show()


# Training
W, n_miss_list = train_ANN(X, sigma, W, epochs, lr, theta)

# After training:
plot_results_ANN(X, sigma, W)



