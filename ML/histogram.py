import os
from numpy import genfromtxt
import matplotlib.pyplot as plt
from analyze import analyze_dataset


def genhistogram(my_data):
    k=1
    for col in range(my_data.shape[1]):
        a=my_data[:,col]
        plt.subplot(abs(my_data.shape[1]/2)+1,2,k)
        plt.hist(a, bins='auto')  # arguments are passed to np.histogram
        #plt.title("Histogram with 'auto' bins")
        k+=1
    plt.show()
