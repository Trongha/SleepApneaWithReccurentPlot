import os
import numpy as np
import config as config


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def readDataPreProcess(dataFile, labelFile):
    trainDataOrigin = np.load(dataFile, allow_pickle=True)
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


def getLabel(labels):
    '''
    :param labels:
    :return: APNEA_LABEL if number of apnea > 50%, else return NORMAL_LABEL
    '''
    unique, counts = np.unique(labels, return_counts=True)
    dictUnique = dict(zip(unique, counts))
    apnea = config.APNEA_LABEL
    normal = config.NORMAL_LABEL

    if apnea in dictUnique.keys() and dictUnique[apnea] >= 0.5 * len(labels):
        # print('label : apnea')
        return apnea
    return normal
