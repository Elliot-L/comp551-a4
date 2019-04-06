from scipy.sparse import csr_matrix

def covert(arr):
    return arr.todense()

A = csr_matrix([[1, 0, 2], [0, 3, 0]])
print(covert(A))
