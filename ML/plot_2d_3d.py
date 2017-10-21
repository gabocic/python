#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_3d(elements_list,dimensions):

    if dimensions == 2:
        fig,ax = plt.subplots()
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # <-- 3D
    else:
        return

    for element in elements_list:
        if element['type'] == 'dot':
            ax.scatter(*element['value'], color=element['color'], marker=element['marker'],s=element['size'])
        elif element['type'] == 'blob':
            #ax.scatter(*element['value'], color=element['color'])
            if dimensions == 2:
                ax.scatter(element['value'][:,0],element['value'][:,1], color=element['color'])
            elif dimensions == 3:
                ax.scatter(element['value'][:,0],element['value'][:,1],element['value'][:,2], color=element['color'])
        elif element['type'] == 'line':
            ax.plot(*element['value'],'-', color=element['color'])

    # SubTitle
    fig.suptitle("Elements", fontsize=10)

    ## Graph and axis formatting
    ax.set_aspect('equal')
    ax.grid(True, which='both')

    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    #ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    plt.show()
