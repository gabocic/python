#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.preprocessing import scale

def sklearn_scale(data):

    # Center to the mean and component wise scale to unit variance.
    scaleddata = scale(data)

    return scaleddata
