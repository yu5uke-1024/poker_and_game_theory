import numpy as np

a = [ np.array([1,2]), np.array([1,2,3]), np.array([1,2]), np.array([2,2,3])]
a_hashable = map(tuple, a)

print(len(set(a_hashable)))
