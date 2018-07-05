def cost_gradient(y, x, W, b):
#     Calculate the the argument of sigmoid
    z = np.dot(x, W) + b
#     Calculate the derivative of the cost function wrt z
    DcDz = y-sigmoid(z)
#     Calculate the derivatives wrt each parameter
    DcDW = -np.dot(DcDz, x)/len(y)
    DcDb = -np.sum(DcDz)/len(y)
    
    return DcDW, DcDb
