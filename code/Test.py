import numpy as np
arr1 = np.array(np.random.rand(4))
arr2 = np.array(np.random.rand(3,4))

print("Array 1:", arr1.shape)
print("Array 2:", arr2)

print("Shape of min:", np.minimum(arr1, arr2).shape)

print("Minimum:", np.minimum(arr1, arr2))

mini = []
count = 0
for i in range(arr2.shape[0]):
    mini_t = list(np.minimum(arr1, arr2[i, :]))
    mini.append(mini_t)

print("Minimum with for loop:", mini)