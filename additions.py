# Functions to evaluate model
# Chris Finlay
# SKA SA
# cfinlay@ska.ac.za

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
#     Calculate the sigmoid of the argument z and return it
    sig = 1./(1+np.exp(-z))
    return sig

def cost(y, x, W, b):
#     Calculate the argument of the sigmoid
    z = np.dot(x, W) + b
    
#     Calculate the cost and return it
    C = -np.sum(y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z)))/len(y)
    
    return C

def contour_plot(y, x, W, b, W_opt=-5, W_span=6, b_opt=-0.35, b_span=5):

    ww = np.linspace(W_opt-W_span, W_opt+W_span, 100) 
    bb = np.linspace(b_opt-b_span, b_opt+b_span, 100)
    cc = np.zeros((len(bb), len(ww)))
    for i in range(cc.shape[0]):
        for j in range(cc.shape[1]):
            cc[i,j] = cost(y, x, np.array([ww[j]]), bb[i])
            
    plt.contour(cc, extent=(ww[0], ww[-1], bb[0], bb[-1]))
    plt.plot(W, b, 'o-')
    plt.plot(W[-1], b[-1], 'ro')
    plt.xlabel("W")
    plt.ylabel("b")
    plt.colorbar()


def training_plots(W, b, C):
    fig = plt.figure(figsize=(15,5))
    ax = fig.subplots(nrows=1, ncols=3)
    titles = ["Cost (C)", "Weights (W)", "Bias (b)"]
    arrays = [C, W, b]
    for i in range(3):
        ax[i].plot(arrays[i])
        ax[i].set_title(titles[i])
        ax[i].set_xlabel("Iterations")

def accuracy(y_test, x_test, W, b, threshold):
    z = np.dot(x_test, W) + b
    p = sigmoid(z)
    predictions = np.where(p>threshold, 1, 0)
    acc = np.mean(np.where(predictions==y_test, 1.0, 0.0))*100
    return acc, predictions

def ROC(y_test, x_test, W, b):
    fpr = np.zeros(101)
    tpr = np.zeros(101)
    for i in range(101):
        acc, pred = accuracy(y_test, x_test, W, b, i/100)
        fpr[i] = np.sum(np.where(((pred!=y_test) & (pred==1)), 1.0, 0.0))/(len(y_test)-np.sum(y_test))
        tpr[i] = np.sum(np.where(((pred==y_test) & (pred==1)), 1.0, 0.0))/np.sum(y_test)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operator Characteristic (ROC) Curve")

    return fpr, tpr


from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(test_labels, predictions,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    classes = ['Exoplanet', 'RRLyrae']
    cm= confusion_matrix(test_labels, predictions)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


