import numpy as np
import config as config


def getBinaryByRow(row, dotRate):
    print('origin: ', row)
    rowClone = np.array(row)
    epsilon = rowClone[int(dotRate*len(row))]
    return (rowClone < epsilon) + 0

a = [50, 1, 2, 7, 1, 3, 2, 10, 11, 100, 0]
print(a)

print(getBinaryByRow(a))

numTrain = int(l / 3 * 2)

train = a[0:numTrain]
test = a[numTrain:]

print('Hello World')

dataFile = config.FOLDER_SAVE_RQA + 'rqa.npy'
infoFile = config.FOLDER_SAVE_RQA + 'info.npy'
rqa = np.load(dataFile, allow_pickle=True)
info = np.load(infoFile, allow_pickle=True)
labels = [i[3] for i in info]
for i, label in enumerate(labels):
    if label == config.APNEA_LABEL:
        print('apnea ', i, label)
print('stop')

# print(a, b)
# print(np.append(a, b))
