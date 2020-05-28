import numpy as np
from src import config as config
from src import MyUtil as myUtil
from tqdm import tqdm
from src import RecurrentPlot as rp
from src import RecurrenceQuantificationAnalysis as rqa

print('test load rp')
recordNames = config.NAME_OF_RECORD
numTrain = config.NUMBER_OF_TRAIN_RECORD

print('load train . . .')
trainRp = []
trainLabel = []
for iRecord in tqdm(range(0, numTrain)):
    rpBinary, label, _ = myUtil.readRpBinary(recordNames[iRecord], 'train')
    trainRp = np.append(trainRp, rpBinary, axis=0) if len(trainRp) > 0 else rpBinary
    trainLabel = np.append(trainLabel, label, axis=0) if len(trainLabel) > 0 else label
print('trainRp: ', trainRp.shape)
print('trainLabel: ', trainLabel.shape)

print('load test . . .')
testRp = []
testLabel = []
for iRecord in tqdm(range(numTrain, len(recordNames))):
    rpBinary, label, _ = myUtil.readRpBinary(recordNames[iRecord], 'test')
    testRp = np.append(testRp, rpBinary, axis=0) if len(testRp) > 0 else rpBinary
    testLabel = np.append(testLabel, label, axis=0) if len(testLabel) > 0 else label
print('done load data')

print('testRp: ', testRp.shape)
print('testLabel: ', testLabel.shape)

print('stop')

# print(a, b)
# print(np.append(a, b))
