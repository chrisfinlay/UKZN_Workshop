def cost(y, x, W, b):
#     Calculate the argument of the sigmoid
    z = np.dot(x, W) + b
    
#     Calculate the cost and return it
    C = -np.sum(y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z)))/len(y)
    
    return C
