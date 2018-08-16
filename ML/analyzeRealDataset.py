from numpy import genfromtxt
from analyze import analyze_dataset

#my_data = genfromtxt('/home/gabriel/Desktop/segmentation.csv', delimiter=',')
#my_data = genfromtxt('/home/gabriel/Downloads/optdigits.tra', delimiter=',')
#my_data = genfromtxt('/home/gabriel/Desktop/page-blocks.data', delimiter=',')
#my_data = genfromtxt('/home/gabriel/Downloads/phplE7q6h.csv', delimiter=',')
my_data = genfromtxt('/home/gabriel/Downloads/datasets/house16H.dat', delimiter=',',max_rows=2000)
print(my_data.shape)
output = analyze_dataset(data=my_data,debug=1,plot=1,load_from_file=None)
print(output)
