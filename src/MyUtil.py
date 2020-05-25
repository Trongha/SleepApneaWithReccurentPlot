import os
from typing import Union, Iterable

import numpy as np
from numpy.core._multiarray_umath import ndarray

from src import config as config


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def readDataPreProcess(dataFile, labelFile):
    trainDataOrigin: Union[ndarray, Iterable, int, float, tuple, dict] = np.load(dataFile, allow_pickle=True)
    labelOrigin = np.load(labelFile, allow_pickle=True)

    rriData = []  # [[record1], [record2], ...]
    label = []  # [[listLabels1], [listLabels2], ...]
    # ===> rriData[i][j] has label[i][j]

    curId = None
    curContainer = []
    curListLabel = []
    for i, originData in enumerate(trainDataOrigin):
        rri = originData[0]
        id = originData[1]
        if id == curId:
            curContainer = np.append(curContainer, np.array(rri))
            curListLabel += [labelOrigin[i] for numberOf in range(len(rri))]
        else:
            if curId is not None:
                rriData.append(curContainer)
                label.append(curListLabel)

            curId = id
            curContainer = rri
            curListLabel = [labelOrigin[i] for numberOf in range(len(rri))]
            print("curId: ", curId)

    rriData.append(curContainer)
    label.append(curListLabel)
    return rriData, label


def getLabelAndInfo(recordName='a01'):
    labelFile = config.getFileSaveLabel(recordName)
    infoFile = config.getFileSaveInfo(recordName)
    label = np.load(labelFile, allow_pickle=True)
    info = np.load(infoFile, allow_pickle=True)
    return label, info


def readRpBinary(recordName='a01'):
    # todo: use getFileName
    # data = np.load(fileData)
    dataFile = config.getFileSaveRp(recordName)
    data = np.load(dataFile, allow_pickle=True)
    label, info = getLabelAndInfo(recordName)
    return data, label, info


def getLabel(labels):
    '''
    :param labels:
    :return: APNEA_LABEL if number of apnea > 50%, else return NORMAL_LABEL
    '''
    unique, counts = np.unique(labels, return_counts=True)
    dictUnique = dict(zip(unique, counts))
    apnea = config.APNEA_LABEL
    normal = config.NORMAL_LABEL
    minPercentApnea = config.MIN_PERCENT_APNEA
    if apnea in dictUnique.keys() and dictUnique[apnea] >= minPercentApnea * len(labels):
        # print('label : apnea')
        return apnea
    return normal


if __name__ == '__main__':
    print('MyUtil run main')
    recordNames = config.NAME_OF_RECORD
    data, label, info = readRpBinary(recordNames[2])
    print('data:', data.shape)
    print('label:', label.shape)
    print('info:', info.shape)
    from src import RecurrentPlot as rp
    for i in [0, 10, 19]:
        listDot = data[i]
        thisLabel = label[i]
        title = 'N-' if thisLabel == config.NORMAL_LABEL else 'A-'
        title += 'record-{}.start-{}.end-{}'.format(2, info[i][1], info[i][2])
        rpBinary = rp.getRpBinaryFromListDot(listDot)
        rp.crossRecurrencePlots(title, rpBinary, myTitle=title, showPlot=True)

    print('finish code')

