print('Hello World')
import numpy as np



a = [1, 2,3 , 1, 3, 2, 2, 2, 1, 2]
b = [2, 1, 4]

a = np.array(a)
b = np.array(b)
unique, counts = np.unique(a, return_counts=True)
cc = dict(zip(unique, counts))
print('cc', cc)
for i in range(8):
    print(i)
    if i in cc.keys():
        print(cc[i])
    else:
        print('okok')
print(cc[100] is None)
print(unique)
print(counts)

# print(a, b)
# print(np.append(a, b))