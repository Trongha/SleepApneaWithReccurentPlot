import matplotlib.pyplot as plt
import numpy as np
import config
import json

import src.MyUtil as myUtil

# res = '../res/train/'
res = '../output/'
trainDataFile = res + 'my_train_input.npy'
trainLabelFile = res + 'my_train_label.npy'


def convert2StatePhase(timeSeries, dim, tau, returnType='array'):
    # v: vecto
    # dim là số chiều (số phần tử)
    # tau là bước nhảy
    # returnType: kiểu trả về
    #	array: trả về python array
    #	np.array: trả về mảng numpy
    if returnType == 'array':
        return [timeSeries[start: start + (dim - 1) * tau + 1: tau]
                for start in range(len(timeSeries) - (dim - 1) * tau)]
    if returnType == 'np.array':
        import numpy as np
        return np.array([timeSeries[start: start + (dim - 1) * tau + 1: tau]
                         for start in range(len(timeSeries) - (dim - 1) * tau)])


def makeRpMatrix(timeSeries, dim=5, tau=2, epsilon=0.09, distNorm=2):
    # tách statephases
    statePhase = convert2StatePhase(timeSeries, dim, tau, 'np.array')

    from scipy.spatial.distance import cdist
    # rDist là ma trận khoảng cách
    # cdist là hàm trong scipy.spatial.distance.cdist
    # minkowski là cách tính
    # p là norm
    # y là train: đánh số từ trên xuống dưới
    # x là test
    rDist = cdist(statePhase, statePhase, 'minkowski', p=distNorm)

    import numpy as np
    rBinary = np.array((rDist < epsilon) + 0)

    return rBinary


# vẽ biểu đồ chấm từ mảng x và mảng y
def scatterGraph(windowTitle, dataX, dataY, dotSize=0, myTitle='prettyGirl', labelX='xxxxx', labelY='yyyyy'):
    f = plt.figure(windowTitle)
    plt.scatter(dataX, dataY, s=dotSize)
    plt.title(myTitle)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    return f


# vẽ biểu đồ crp từ ma trận 01
def crossRecurrencePlots(windowTitle, dataMatrixBinary, keyDot=1, dotSize=1, myTitle='prettyGirl', labelX='xxxxx',
                         labelY='yyyyy', showPlot=False, pathSaveFigure=None):
    dataX = []
    dataY = []
    hightOfData = len(dataMatrixBinary)

    print("crossRecurrencePlots()_len: ", hightOfData)
    for y in range(hightOfData):
        for x in range(len(dataMatrixBinary[y])):
            if dataMatrixBinary[y][x] == keyDot:
                dataX.append(x)
                # append hight-y nếu muốn vẽ đồ thị đúng chiều như lưu trong ma trận
                dataY.append(hightOfData - y - 1)
            # vẽ trục y từ dưới lên
            # dataY.append(y);

    figure = scatterGraph(windowTitle, dataX, dataY, dotSize, myTitle, labelX, labelY)
    if showPlot:
        plt.show()
    if pathSaveFigure is not None:
        plt.savefig(pathSaveFigure, dpi=200)
    return figure


def convertSetNumber(Set, minOfSet=0, maxOfSet=0, newMinOfSet=0, newMaxOfSet=1):
    if minOfSet == 0:
        minOfSet = min(Set)
    if maxOfSet == 0:
        maxOfSet = max(Set)

    print("min: ", minOfSet)
    print("max: ", maxOfSet)

    if maxOfSet == minOfSet:
        ratio = 0
    else:
        ratio = (newMaxOfSet - newMinOfSet) / (maxOfSet - minOfSet)
    return [((x - minOfSet) * ratio + newMinOfSet) for x in Set]


# ============================== read data =================================#
def readData(dataFile, labelFile):
    trainDataOrigin = np.load(dataFile, allow_pickle=True)
    labelOrigin = np.load(labelFile, allow_pickle=True)

    trainData = []
    label = []

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
                trainData.append(curContainer)
                label.append(curListLabel)

            curId = id
            curContainer = rri
            curListLabel = [labelOrigin[i] for numberOf in range(len(rri))]
            print("curId: ", curId)

    trainData.append(curContainer)
    label.append(curListLabel)
    return trainData, label


def getLabel(labels):
    unique, counts = np.unique(labels, return_counts=True)
    dictUnique = dict(zip(unique, counts))
    if 1 in dictUnique.keys():
        print('unique has apnea', dictUnique)
    if 1 in dictUnique.keys() and dictUnique[1] >= 0.5 * len(labels):
        print('label : apnea')
        return 1
    return 0.0


# ============================================================================#

if __name__ == '__main__':
    print("RecurrentPlot.py run main")
    # ============================== Load Config =================================#
    winSize = config.RR_PER_RECURRENCE_PLOTS
    dim = config.DIMENSION
    tau = config.TAU
    e = config.EPSILON
    disNorm = config.DISTANCE_NORM
    # ============================================================================#

    trainData, label = readData(trainDataFile, trainLabelFile)
    print(len(trainData))
    print(len(label))
    myUtil.createFolder(config.SAVE_RP_FOLDER)
    for i_data, data in enumerate(trainData):
        print('len of item ', i_data, len(data), len(label[i_data]))
        if len(data) > winSize:
            for start in range(len(data) - winSize):
                end = start + winSize
                timeSeries = data[start:end]
                thisLabel = getLabel(label[i_data][start:end])
                timeSeries = convertSetNumber(timeSeries)
                binaryMatrix = makeRpMatrix(timeSeries, dim, tau, e, disNorm)
                # print(binaryMatrix)

                title = 'N' if thisLabel == config.NORMAL_LABEL else 'A'
                title += '---record-{}.start-{}.end-{}'.format(i_data, start, end)
                pathSave = config.SAVE_RP_FOLDER + title + config.IMG_SUFFIX
                x = crossRecurrencePlots(windowTitle=title, dataMatrixBinary=binaryMatrix,
                                         myTitle=title, pathSaveFigure=pathSave)
                # plt.show()
        else:
            print(" len of data < winSize({})".format(winSize))
