import numpy as np
from numpy.linalg import inv
p1 = np.array([3, 3, 3])
p2 = np.array([1, 2, 3])
p3 = np.array([0, 0, 1])
A = np.array([[1,1], [1,0], [1,0]])
A_T = A.transpose()
A_square = np.matmul(A_T, A)
A_first = np.matmul(A, A_square)
A_next = np.matmul(A_first, A_T)
p1_cool = np.matmul(A_next, p1)
p2_cool = np.matmul(A_next, p2)
p3_cool = np.matmul(A_next, p3)

print("P1 = " + str(p1_cool))
print("P2 = " + str(p2_cool))
print("P3 = " + str(p3_cool))