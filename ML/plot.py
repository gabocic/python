from plot_2d_3d import plot_2d_3d
import numpy as np 
from numpy.linalg import norm

n_features=3

mean = np.array([2,3,5])
p0 = np.array([6,7,8])
linea = np.array([p0,mean])

theta=2.37
p1 = mean + theta*(p0-mean)

print('Distance from mean to p0:',norm(p0-mean))
print('Distance from mean to p1:',norm(p1-mean))
print('Cociente:',norm(p1-mean)/norm(p0-mean))

element_list=[]
element={'type':'dot','value':mean,'color':'r','marker':'x','size':30}
element_list.append(element)

element={'type':'dot','value':p0,'color':'g','marker':'o','size':30}
element_list.append(element)

element={'type':'line','value':linea.T,'color':'b','marker':'o','size':10}
element_list.append(element)

element={'type':'dot','value':p1,'color':'y','marker':'x','size':30}
element_list.append(element)

plot_2d_3d(element_list,n_features)

