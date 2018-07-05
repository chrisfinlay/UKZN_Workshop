def sigmoid(z):
#     Calculate the sigmoid of the argument z and return it
    sig = 1./(1+np.exp(-z))
    return sig
