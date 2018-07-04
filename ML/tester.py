import numpy as np
import Orange

y_pred = np.array([[1,3,1,2,2]])
y_actual = np.array([1,3,2,2,1])
#y_probs = np.array([[0.1,0.3,0.2],[0.2,0.1,0.8],[0.3,0.4,0.3],[0.2,0.1,0.8],[0.1,0.3,0.2]])
y_probs = np.array([[0.1,0.3,0.2,0.2,0.1]])
#print(len(y_actual))
r = Orange.evaluation.Results(actual=y_actual,predicted=y_pred,nrows=5,nmethods=1,probabilities=y_probs)
