import numpy as np
# myarr = np.array([12,24,51,22], np.int64)  # MEMORY CAN BE MANAGED BY DEVELOPER.
# myarr[1] = 45
# print(myarr)
# print(myarr.dtype)
# print(myarr.shape)

x = [[1,2,3]
    ,[4,5,6]
    ,[7,1,0]]
arr = np.array(x)
# print(arr.T)  # TRANSPOSE OF A MATRIX
# print(arr.sum(axis=0)) # SUM OF COLUMNS
# print(arr.sum(axis=1)) # SUM OF ROWS

# for item in arr.flat:
#     print(item)  # ITERATING THROUGH ALL ELEMENTS OF THE ARRAY

# one = np.array([1,2,3,4,5])
# print(one.argmax())  # RETURNS THE INDEX OF MAXIMUM VALUE IN THE ARRAY
# print(one.argsort())

import sys 
py_ar = [0,4,55,2]
np_ar = np.array(py_ar)
sys.getsizeof(py_ar)  # SIZE OF PYTHON LIST IN BYTES
np.ar.itemsize * np_ar.size  # SIZE OF NUMPY ARRAY IN BYTES.