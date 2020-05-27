import wfdb
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import os

from src import config as config
from src import MyUtil as myUtil

FS = 100.0

outPath = '../res/dataPreProcess/'
dataPath = '../res/origin/'
trainDataName = config.NAME_OF_RECORD


# Return the amplitude (max - min) of the RRi series: biên độ
def get_qrs_amp(ecg, qrs):
    interval = int(FS * 0.250)
    qrs_amp = []

    for index in range(len(qrs)):
        curr_qrs = qrs[index]
        amp = np.max(ecg[curr_qrs - interval:curr_qrs + interval])
        # thisSignal = ecg[curr_qrs-interval:curr_qrs+interval]
        # plt.plot(thisSignal)
        # plt.show()
        qrs_amp.append(amp)

    return qrs_amp


MARGIN = 10
MAX_HR = 300.0
MIN_HR = 20.0
MIN_RRI = 1.0 / (MAX_HR / 60.0) * 1000
MAX_RRI = 1.0 / (MIN_HR / 60.0) * 1000
train_input_array = []
train_label_array = []

myUtil.createFolder(config.PATH_RRI)
for recordIndex, recordName in enumerate(trainDataName):
    print(recordName)
    contentFileTxt = ""
    numberOfLabel = len(wfdb.rdann(os.path.join(dataPath, trainDataName[recordIndex]), 'apn').symbol)
    signals, fields = wfdb.rdsamp(os.path.join(dataPath, trainDataName[recordIndex]))
    for index in tqdm(range(1, numberOfLabel)):
        sampFrom = index * 60 * FS  # 60 seconds
        sampTo = sampFrom + 60 * FS  # 60 seconds

        # from -> to: 80 seconds
        qrsAnn = wfdb.rdann(dataPath + trainDataName[recordIndex], 'qrs', sampfrom=sampFrom,
                            sampto=sampTo).sample
        # print(qrsAnn[:55], qrsAnn[-55:])
        # qrs_ann là thời gian

        # lấy label của dữ liệu apnea
        # from -> to: 6000đv - 1đv
        apnAnn = wfdb.rdann(dataPath + trainDataName[recordIndex], 'apn', sampfrom=sampFrom,
                            sampto=sampTo - 1).symbol

        # diff: out[i] = a[i+1] - a[i] -> Tinh khoang RR, rri la chuoi cac gia tri RR
        # print('\nlen = ', qrsAnn[-1] - qrsAnn[0])
        rri = np.diff(qrsAnn)
        # print("sum rri: ", np.sum(rri))
        rriBySec = rri.astype('float') / FS
        # print("sum rri2: ", np.sum(rriBySec))

        if apnAnn[0] == 'N':  # Normal
            label = config.NORMAL_LABEL
        elif apnAnn[0] == 'A':  # Apnea
            label = config.APNEA_LABEL
        else:
            label = config.NONE_LABEL

        # if index in [1, 2, 3, 4, 5, numberOfLabel - 1]:
        #     print(dataIndex, rriBySec)
        train_input_array.append([rriBySec, recordIndex])
        train_label_array.append(label)

        ########## write txt #########
        if config.NORMAL_LABEL == label:
            contentFileTxt += 'N, '
        elif config.APNEA_LABEL == label:
            contentFileTxt += 'A, '
        else:
            contentFileTxt += 'X, '
        contentFileTxt += '\n'

    # ------------------ End preprocess 1 record ------------------
    fileTxt = config.getFileTxtRri(recordName)
    print('write to file ', fileTxt)
    file = open(fileTxt, 'w')
    file.write(contentFileTxt)
    file.close()
    print('done write to file')

np.save(config.FILE_RRI_NPY, train_input_array)
np.save(config.FILE_RRI_LABEL, train_label_array)
