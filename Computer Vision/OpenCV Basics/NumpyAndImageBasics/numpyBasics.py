import numpy as np

mylist = [1, 2, 3]

myarray = np.array(mylist)

np.arange(0, 10, 2)

np.zeros(shape=(5, 5))

np.ones(shape=(2, 4))

np.random.seed(101)
arr = np.random.randint(0, 100, 10)
arr2 = np.random.randint(0, 100, 10)

arr.max()
arr.argmax()  # return index of max
arr.argmin()  # return index of min
arr.mean()  # return average
# arr.shape()  # returns shape
arr.reshape((2, 5))  # changes the shape
# to change the shape new shape must fit exactly len(arr) elements
mat = np.arange(0, 100).reshape(10, 10)
# to access element :
row, col = 1, 2
ele = mat[row, col]
# slicing
# fist all elements of col, reshape to look like original
col_ele = mat[:, 1].reshape(10, 1)
# to slice in just a box:
box = mat[0:3, 0:4]
# copy constructor
mynewmat = mat.copy()
# if done like this: "mynewmat = mat" then it's a reference call
# and changing mynewmat changes mat
mynewmat[1, 1] = 99

print(mynewmat)