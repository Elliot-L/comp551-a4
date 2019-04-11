import numpy as np

def average(arr):
    avg = 0.0
    dimensions = arr.shape
    if dimensions[0] == 1:
        return 0
    for (x, y), value in np.ndenumerate(arr):
        if(x == int(dimensions[0]/2)) and (y == int(dimensions[1]/2)):
            continue
        else:
            avg = avg+ value
    avg = avg/(pow(dimensions[0],2)-1)
    return avg


x = np.array([[1,2,3],[6,7,8],[7,8,9]])
print(average(x))