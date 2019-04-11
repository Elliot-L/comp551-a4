import numpy as np
class Data:
   def __init__(self,array,size):
       self.start= 0
       self.idx1 = 0
       self.idx2 = size
       self.data = array
       self.max = size
       self.size = size
   
   def __iter__(self):
       return self

   def __next__(self):
       if self.idx2 * self.max > pow(self.data.shape[0],2):
            raise StopIteration
       elif self.idx2 > self.data.shape[0]:
           self.start += 1
           self.max += 1
           self.idx1 = 1
           self.idx2 = self.size +1
           return self.data[self.start:self.max, self.idx1 - 1:self.idx2 - 1]
       else:
           self.idx1 += 1
           self.idx2 += 1
           return self.data[self.start:self.max,self.idx1-1:self.idx2-1]


arr = np.matrix([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]])
b = Data(arr,3)
print(b.__next__())
