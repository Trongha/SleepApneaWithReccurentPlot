import numpy as np
from src import config as config
from src import MyUtil as myUtil
from tqdm import tqdm
from src import RecurrentPlot as rp
from src import RecurrenceQuantificationAnalysis as rqa

recordNames = config.NAME_OF_RECORD
numTrain = config.NUMBER_OF_TRAIN_RECORD
trainRqa = []
trainLabel = []

for iRecord in range(0, numTrain):
    rqa, label, _ = myUtil.loadRqa(recordNames[iRecord], 'train')
    trainRqa = np.append(trainRqa, rqa, axis=0) if len(trainRqa) > 0 else rqa
    trainLabel = np.append(trainLabel, label, axis=0) if len(trainLabel) > 0 else label

testRqa = []
testLabel = []
for iRecord in range(numTrain, len(recordNames)):
    rqa, label, _ = myUtil.loadRqa(recordNames[iRecord], 'test')
    testRqa = np.append(testRqa, rqa, axis=0) if len(testRqa) > 0 else rqa
    testLabel = np.append(testLabel, label, axis=0) if len(testLabel) > 0 else label
print('done load data')

print('stop')

# print(a, b)
# print(np.append(a, b))
