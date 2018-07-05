def gradient_descent(y, x, Wi, bi, alpha, steps):
#     Create arrays for the weights, bias and cost
    W = np.zeros((steps+1, len(Wi)))
    b = np.zeros(steps+1)
    C = np.zeros(steps+1)
    
#   Populate the first entry for each array  
    W[0] = Wi
    b[0] = bi
    C[0] = cost(y, x, Wi, bi)
    
#     Perform gradient descent
    for i in range(steps):
        DW, Db = cost_gradient(y, x, W[i], b[i])
        
        W[i+1] = W[i] - alpha*DW
        b[i+1] = b[i] - alpha*Db
        
        C[i+1] = cost(y, x, W[i+1], b[i+1])
        
    return W, b, C
