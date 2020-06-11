import wfdb
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

from src import config
from src import MyUtil as myUtil
from src import RecurrentPlot as rp
from src import RecurrenceQuantificationAnalysis as rqa

FS = 100.0

outPath = '../res/dataPreProcess/'
dataPath = '../res/origin/'
trainDataName = config.NAME_OF_RECORD


# Return the amplitude (max - min) of the RRi series: biên độ
def getListMomentOfR(ecg, qrs):
    interval = int(FS * 0.250)
    rWave = []

    for index in range(len(qrs)):
        curQrs = qrs[index]
        # print(curQrs)
        startTime = curQrs - interval  # unit: 10ms
        endTime = curQrs + interval  # unit: 10ms
        ecgHaveR = ecg[startTime:endTime]
        amp = np.max(ecgHaveR)  # đỉnh sóng
        rMoment = startTime + np.where(ecgHaveR == amp)[0][0]  # use [0][0] because np.where return 1 tuple has len == 1
        # thisSignal = ecg[curQrs-interval:curQrs+interval]
        # plt.plot(thisSignal)
        # plt.show()
        rWave.append(rMoment)
    return rWave


MARGIN = 10
MAX_HR = 300.0
MIN_HR = 20.0
MIN_RRI = 1.0 / (MAX_HR / 60.0) * 1000
MAX_RRI = 1.0 / (MIN_HR / 60.0) * 1000
trainInputArray = []
trainLabelArray = []
exceptionRriArray = []
minuteBiasArray = []

myUtil.createFolder(config.PATH_RRI)

for recordIndex, recordName in enumerate(trainDataName):
    print(recordName)
    contentFileTxt = ""
    numberOfLabel = len(wfdb.rdann(os.path.join(dataPath, trainDataName[recordIndex]), 'apn').symbol)
    signals, fields = wfdb.rdsamp(os.path.join(dataPath, trainDataName[recordIndex]))

    lastQrsOfPreMinute = None
    for index in tqdm(range(1, numberOfLabel)):
        sampFrom = (index * 60 * FS) if lastQrsOfPreMinute is None else lastQrsOfPreMinute
        sampTo = (index + 1) * 60 * FS  # 60 seconds
        # from -> to: 80 seconds
        qrsAnn = wfdb.rdann(dataPath + trainDataName[recordIndex], 'qrs', sampfrom=sampFrom,
                            sampto=sampTo).sample

        # ecg = signals[qrsAnn]
        # plt.plot(ecg)
        # plt.show()
        lastQrsOfPreMinute = qrsAnn[-1] if len(qrsAnn) > 0 else None

        # print(qrsAnn[:55], qrsAnn[-55:])
        # qrsAnn là thời gian

        # lấy label của dữ liệu apnea
        # from -> to: 6000đv - 1đv
        apnAnn = wfdb.rdann(dataPath + trainDataName[recordIndex], 'apn', sampfrom=sampFrom,
                            sampto=sampTo - 1).symbol

        # Lấy các thời điểm của đỉnh R theo đơn vị 10ms
        rMoments = getListMomentOfR(signals, qrsAnn)

        # diff: out[i] = a[i+1] - a[i] -> Tinh khoang RR, rri la chuoi cac gia tri RR
        rri = np.diff(rMoments)  # unit: 10ms
        rriBySec = rri.astype('float') / FS
        # collect bias between one minute with sumRri in one label
        minuteBiasArray.append(abs(config.MINUTE - np.sum(rriBySec)))

        # print("sum rri2: ", np.sum(rriBySec))
        # print('index: ', index, 'rriSec: ', rriBySec)
        if len(rriBySec) == 0:
            print('len rri by sec == 0')
            continue
        if np.max(rriBySec) > config.MAX_RRI_BY_SEC:
            print('\nException recordIndex: {} indexStart: {}, sampFrom: {}'.format(recordIndex, index, sampFrom))
            exceptionRriArray.append(np.max(rriBySec))

        if apnAnn[0] == 'N':  # Normal
            label = config.NORMAL_LABEL
        elif apnAnn[0] == 'A':  # Apnea
            label = config.APNEA_LABEL
        else:
            label = config.NONE_LABEL

        # if index in [1, 2, 3, 4, 5, numberOfLabel - 1]:
        #     print(dataIndex, rriBySec)
        trainInputArray.append([rriBySec, recordIndex])
        trainLabelArray.append(label)

        ########## write txt #########
        if config.NORMAL_LABEL == label:
            contentFileTxt += 'N, '
        elif config.APNEA_LABEL == label:
            contentFileTxt += 'A, '
        else:
            contentFileTxt += 'X, '
        for item in rriBySec:
            contentFileTxt += ',' + str(item)
        contentFileTxt += '\n'

    # ------------------ End preprocess 1 record ------------------
    fileTxt = config.getFileTxtRri(recordName)
    print('write to file ', fileTxt)
    file = open(fileTxt, 'w')
    file.write(contentFileTxt)
    file.close()
    print('done write to file')

np.save(config.FILE_RRI_NPY, trainInputArray)
np.save(config.FILE_RRI_LABEL, trainLabelArray)
print('done save rri\n'
      'save exception and info')

fileSaveBias = config.PATH_RRI + 'bias' + '.txt'
file = open(fileSaveBias, 'w')
for bias in minuteBiasArray:
    file.write(str(bias))
file.close()

fileSaveException = config.PATH_RRI + 'exception' + '.txt'
file = open(fileSaveException, 'w')
for exception in exceptionRriArray:
    file.write(str(exception))
file.close()
