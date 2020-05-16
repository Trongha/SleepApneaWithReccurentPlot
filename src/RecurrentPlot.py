import matplotlib.pyplot as plt
import numpy as np
import json

configFile = '../config.json'

res = '../res/train/'
res = '../output/'
trainDataFile = res + 'my_train_input.npy'
trainLabelFile = res + 'my_train_label.npy'


def convert2StatePhase(v, dim, tau, returnType='array'):
    # v: vecto
    # dim là số chiều (số phần tử)
    # tau là bước nhảy
    # returnType: kiểu trả về
    #	array: trả về python array
    #	np.array: trả về mảng numpy
    if returnType == 'array':
        return [v[start: start + (dim - 1) * tau + 1: tau] for start in range(len(v) - (dim - 1) * tau)]
    if returnType == 'np.array':
        import numpy as np
        return np.array([v[start: start + (dim - 1) * tau + 1: tau] for start in range(len(v) - (dim - 1) * tau)])


def makeRpMatrix(TimeSeries, dim=5, tau=2, epsilon=0.09, distNorm=1):
    # tách statephases
    statePhase = convert2StatePhase(TimeSeries, dim, tau, 'np.array')

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
    # plt.show()
    return f


# vẽ biểu đồ crp từ ma trận 01
def crossRecurrencePlots(windowTitle, dataMatrixBinary, keyDot=1, dotSize=1, myTitle='prettyGirl', labelX='xxxxx',
                         labelY='yyyyy'):
    dataX = []
    dataY = []
    hightOfData = len(dataMatrixBinary);

    print("crossRecurrencePlots()_len: ", hightOfData)
    for y in range(hightOfData):
        for x in range(len(dataMatrixBinary[y])):
            if dataMatrixBinary[y][x] == keyDot:
                dataX.append(x)
                # append hight-y nếu muốn vẽ đồ thị đúng chiều như lưu trong ma trận
                dataY.append(hightOfData - y - 1)
            # vẽ trục y từ dưới lên
            # dataY.append(y);

    return scatterGraph(windowTitle, dataX, dataY, dotSize, myTitle, labelX, labelY)


def convertSetNumber(Set, minOfSet=0, maxOfSet=0, newMinOfSet=0, newMaxOfSet=1):
    if (minOfSet == 0):
        minOfSet = min(Set)
    if (maxOfSet == 0):
        maxOfSet = max(Set)

    print("min: ", minOfSet)
    print("max: ", maxOfSet)

    if (maxOfSet == minOfSet):
        ratio = 0
    else:
        ratio = (newMaxOfSet - newMinOfSet) / (maxOfSet - minOfSet)
    return [((x - minOfSet) * ratio + newMinOfSet) for x in Set]


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
            if curId != None:
                trainData.append(curContainer)
                label.append(curListLabel)

            curId = id
            curContainer = rri
            curListLabel = [labelOrigin[i] for numberOf in range(len(rri))]
            print("curId: ", curId)

    trainData.append(curContainer)
    label.append(curListLabel)
    return trainData, label


if __name__ == '__main__':
    print("RecurrentPlot.py run main")
    trainData, label = readData(trainDataFile, trainLabelFile)
    print(trainData)
    print(len(trainData))
    print(label)
    print(len(label))

    for i, data in enumerate(trainData):
        print('len of item ', i, len(data), len(label[i]))

    #
    # trainDataOrigin = np.load(trainDataFile, allow_pickle=True)
    # label = np.load(trainLabelFile, allow_pickle=True)
    # # print(trainDataOrigin[0])
    # print("my")
    # print(trainDataOrigin[1000])
    # print(trainDataOrigin[1000][1])
    # print(trainDataOrigin[0].shape)
    # print(len(trainDataOrigin[0]))
    # print(trainDataOrigin.shape)
    #
    # print("my")
    # rriData = [(data[0] - np.min(data[0])) for data in trainDataOrigin]
    # print('lenData: ', len(rriData))
    # print('lenLabel: ', len(label))
    #
    # with open(configFile) as f:
    #     config = json.load(f)
    # print('config: ', config)
    # e = config["epsilon"]
    # lamb = config["lambda"]
    # disNorm = config["distanceNorm"]
    # dim = config['dim']
    # tau = config["tau"]
    # numPoint = config["numPointPerTimeSeries"]
    #
    # # for i, term in enumerate(rriData):
    # #     print("{} len: {} label: {}".format(i, len(term), label[i]))
    #
    # for i in range(10):
    #     # print('sum: ', rriData[i].sum())
    #     rri = convertSetNumber(rriData[i])
    #     binaryMatrix = makeRpMatrix(rri, dim, tau, e, disNorm)
    #     # print(binaryMatrix)
    #     x = crossRecurrencePlots("testPlot", binaryMatrix)
    #     # plt.interactive(True)
    #     plt.show()
