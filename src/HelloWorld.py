import numpy as np
from src import config as config
from src import MyUtil as myUtil
from tqdm import tqdm
from src import RecurrentPlot as rp
from src import RecurrenceQuantificationAnalysis as rqa

import time
import datetime
from threading import Thread
from multiprocessing import Process

# allRecordNames = config.NAME_OF_RECORD
# recordNameForTest = ['a02', 'a03', 'a04', 'a05']
# recordNameForTrain = [recordName for recordName in allRecordNames if recordName not in recordNameForTest]
# print('record for train: ', recordNameForTrain)
# print('record for test: ', recordNameForTest)

a = [10, 2, 4, 10, 5, 2, 10, 3, 4, 10, 10]
a = np.array(a)

print(a[4: 7])
myMax = np.max(a[4: 7])
print(myMax)
cc1 = np.where(a == myMax)
print(cc1[0].shape)
print(cc1)
cc = np.where(a[4: 7] == myMax)
# print(cc)

# print('where: ', cc)
# for recordName in recordNames:
#     label, _ = myUtil.getLabelAndInfo(recordName, 'test')
#     unique, count = np.unique(label, return_counts=True)
#     print('recordName: ', recordName, ' --- label unique: ', dict(zip(unique, count)))

#
# for i in range(0, 10):
#     startTime = time.time()
#     print(str(startTime))
#     print('=======> Load cluster ', i, ' . . . ')
#
#     recordNames = config.NAME_OF_RECORD[1:5]
#     listRp, listLabel, _ = myUtil.loadRpByCluster(10, 2, listRecordNames=recordNames, type='train')
#     print(listRp.shape)
#     print(listLabel.shape)
#     endTime = time.time()
#     print('duration: ', endTime-startTime)
