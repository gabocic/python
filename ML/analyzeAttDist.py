import os
import  numpy as np
from numpy import genfromtxt
#from analyze import analyze_dataset
import matplotlib.pyplot as plt
from analyze import analyze_dataset


#dsdir = "/home/gabriel/Documents/Datasets/groups"
dsdir = "/home/gabriel/Documents/Datasets/totest"
#dsdir = "/home/gabriel/Documents/Datasets"
#dsdir = "/home/gabriel/Downloads/datasets"
filesindir = os.listdir(dsdir)


i=0
for fila in filesindir:
    if fila.endswith(".dat") or fila.endswith(".txt") or fila.endswith(".csv"):
    #if fila.endswith(".csv"):
        print('')
        print('')
        print('##################')
        print(fila)
        print('##################')
        print('')
        my_data = genfromtxt(dsdir+'/'+fila, delimiter=',',max_rows=4000)
        #print(my_data)
        #my_data = my_data[~np.isnan(my_data)]
        #print(my_data)

        output = analyze_dataset(data=my_data,debug=1,plot=0,load_from_file=None)
        print(output)
        
        if i >= 0:
            k=1
            for col in range(my_data.shape[1]):
                a=my_data[:,col]
                plt.subplot(abs(my_data.shape[1]/2)+1,2,k)
                #plt.hist(a, bins='auto')  # arguments are passed to np.histogram
                #plt.title("Histogram with 'auto' bins")
                k+=1
            #plt.show()
        i+=1
