import numpy as np
import matplotlib.pyplot as plt

def y(x):
    return x**2

def y_prime(x):
    return 2*x

def gradient_descent(Wi, steps, alpha):
    W = np.zeros(steps)
    W[0] = Wi
    for i in range(steps-1):
        W[i+1] = W[i] - alpha*y_prime(W[i])
    return W

def learning_rate(Wi, steps, alpha):
    W = gradient_descent(Wi, steps, alpha)
    x = np.arange(-np.max(np.abs(W)), np.max(np.abs(W)), 0.1)
    plt.plot(W, y(W), "ro-")
    plt.plot(x, y(x), 'b')
    plt.title("Gradient Descent Example")
    plt.xlabel("W")
    plt.ylabel("Cost")
    for i in range(len(W)):
        plt.text(W[i]+0.1, y(W[i])+0.1, str(i+1))
