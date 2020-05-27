import numpy as np
# import json
import src.RecurrentPlot as rp
import src.MyUtil as myUtil
from src import config as config
from tqdm import tqdm


def getRQA(TimeSeries, dim=5, tau=2, epsilon=0.09, lambd=2, distNorm=1,
           typeReturn='array', showCRP=False):
    if len(TimeSeries) > (dim - 1) * tau + 1:
        norm01TimeSeries = rp.convertSetNumber(TimeSeries)

        r_Binary = rp.makeRpMatrix(norm01TimeSeries, dim=dim, tau=tau, epsilon=epsilon, distNorm=distNorm)

        return rqaCalculate(r_Binary, keyDot=1, lambd=lambd, typeReturn=typeReturn, showCRP=showCRP)

    return None


def rqaCalculate(rpBinaryMatrix, keyDot=1, lambd=8, typeReturn='array', showCRP=False):
    import math

    len_rpBinaryMatrix = int(np.size(rpBinaryMatrix[0]))
    N = high_rpBinaryMatrix = int(np.size(rpBinaryMatrix) / len_rpBinaryMatrix)

    # content = []
    # for y in range(high_rpBinaryMatrix):
    #     s = ";".join(str(i) for i in rpBinaryMatrix[y])
    #     s += "\n"
    #     content.append(s)
    # print(content)

    # myCrpFunctions.writeContentToFile('rqaOut.csv', content)

    rr = 0
    det = 0
    lam = 0
    ratio = 0
    averageL = 0
    averageH = 0
    div = 0
    entr = 0
    TT = 0

    # Duyệt các đường cao
    ph, hmax, averageTime1, averageTime2 = getPverticalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix,
                                                                 len_rpBinaryMatrix, keyDot=keyDot)

    # Đếm số đường chéo theo độ dài
    pl, lmax = getPDiagonalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix, len_rpBinaryMatrix, keyDot=keyDot)

    num_L = 0
    num_H = 0
    sum_Well_L = 0
    sum_Well_H = 0
    num_Well_L = 0
    num_Well_H = 0

    sumDot = ph[0]  # Đếm số điểm chấm theo đường thẳng, ph[0] là những đoạn chỉ có 1 chấm

    lmin = lambd - 1

    for i in range(1, N + 1, 1):
        sumDot += (i + 1) * ph[i]

        num_L += pl[i]
        num_H += ph[i]

        if (i >= lmin):
            sum_Well_L += pl[i] * i
            sum_Well_H += ph[i] * i

            num_Well_L += pl[i]
            num_Well_H += ph[i]

    rr = sumDot / (N ** 2)

    if num_L * num_Well_L > 0:
        det = num_Well_L / num_L
        averageL = sum_Well_L / num_Well_L

    if (det > 0):
        ratio = det / rr

    if num_H * num_Well_H > 0:
        lam = num_Well_H / num_H
        averageH = sum_Well_H / num_Well_H

    # ============================== Calc entr =================================#
    # print('lmin: ', lmin)
    for i in range(lmin, N, 1):
        if pl[i] > 0:
            pll = pl[i] / num_Well_L
            entr -= pll * math.log(pll)
    # ==========================================================================#
    if lmax > 0:
        div = 1 / lmax

    if showCRP:
        import matplotlib.pyplot as plt
        x = rp.crossRecurrencePlots("test", rpBinaryMatrix, keyDot=keyDot, dotSize=10)
        plt.show()
        print(rr)
        print(det)
        print(lam)
        print("averageL, averageH", averageL, averageH)
        print(lmax, hmax)
        print(div)
        print(entr)
        print("averageTime1, averageTime2", averageTime1, averageTime2)

    if typeReturn == 'array':
        return [rr, det, lam, ratio, averageL, averageH, lmax, hmax, div, entr, averageTime1, averageTime2]
    if typeReturn == 'dict':
        return {
            "rr": rr,
            "det": det,
            "lam": lam,
            "ratio": ratio,
            "averageL": averageL,
            "averageH": averageH,
            "lmax": lmax,
            "Hmax": hmax,
            "div": div,
            "entr": entr,
            "averageTime1": averageTime1,
            "averageTime2": averageTime2
        }


def getPverticalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix=None, len_rpBinaryMatrix=None, keyDot=1):
    if (high_rpBinaryMatrix is None or len_rpBinaryMatrix is None):
        len_rpBinaryMatrix = int(np.size(rpBinaryMatrix[0]))
        high_rpBinaryMatrix = int(np.size(rpBinaryMatrix) / len_rpBinaryMatrix)
    N = high_rpBinaryMatrix

    ph = [0] * (N + 1)
    sumT1 = 0
    sumT2 = 0
    numT = 0
    Hmax = 0
    averageTime1 = 0
    averageTime2 = 0

    for x in range(len_rpBinaryMatrix):
        start = 0
        length = 0
        numDot = 0
        y = 0
        while y < high_rpBinaryMatrix:
            if keyDot == rpBinaryMatrix[y][x]:
                if y > 0 and numDot > 0:
                    t2now = y - start
                    sumT1 += (t2now - length)
                    sumT2 += t2now
                    numT += 1
                start = y
                while y + 1 < high_rpBinaryMatrix and rpBinaryMatrix[y + 1][x] == keyDot:
                    y += 1

                numDot = y - start + 1
                length = numDot - 1

                ph[length] += 1

                if (length > Hmax):
                    Hmax = length
            y += 1

    if numT > 0:
        averageTime1 = sumT1 / numT
        averageTime2 = sumT2 / numT

    return ph, Hmax, averageTime1, averageTime2


def getPDiagonalLengthDot(rpBinaryMatrix, high_rpBinaryMatrix=None, len_rpBinaryMatrix=None, keyDot=1):
    if high_rpBinaryMatrix is None or len_rpBinaryMatrix is None:
        len_rpBinaryMatrix = int(np.size(rpBinaryMatrix[0]))
        high_rpBinaryMatrix = int(np.size(rpBinaryMatrix) / len_rpBinaryMatrix)
    N = high_rpBinaryMatrix

    pl = [0] * (N + 1)
    lmax = 0
    # Duyet duong cheo
    for index_diagonal in range(-(high_rpBinaryMatrix + 1), len_rpBinaryMatrix + 2, 1):
        offset = index_diagonal
        # ---offset = x - y
        y = -offset if (index_diagonal < 0) else 0

        while y < high_rpBinaryMatrix and y + offset < len_rpBinaryMatrix:
            if rpBinaryMatrix[y][y + offset] == keyDot:
                start = y
                while (y + 1 < high_rpBinaryMatrix
                       and y + 1 + offset < len_rpBinaryMatrix
                       and rpBinaryMatrix[y + 1][y + 1 + offset] == keyDot):
                    y += 1
                numDot = y - start + 1
                if numDot < N:
                    length = numDot - 1
                    pl[length] += 1
                    if length > lmax:
                        lmax = length
                else:
                    pass
                    # print(numDot, "----------------------------------------------")
            y += 1

    return pl, lmax


if __name__ == '__main__':
    print("RecurrenceQuantificationAnalysis.py run main")
    # ============================== Load Config =================================#
    winSize = config.RR_PER_RECURRENCE_PLOTS
    dim = config.DIMENSION
    tau = config.TAU
    e = config.EPSILON
    disNorm = config.DISTANCE_NORM
    myLambda = config.MY_LAMBDA
    # ============================================================================#
    print('rri: ', config.FILE_RRI_NPY)
    rriData, label = myUtil.loadRri(config.FILE_RRI_NPY, config.FILE_RRI_LABEL)

    print(len(rriData))
    print(len(label))

    myUtil.createFolder(config.PATH_RQA)
    fileData = config.PATH_RQA + 'rqa.npy'
    fileInfo = config.PATH_RQA + 'info.npy'  # [[label, idRecord, start, end], [], ...]
    allRqa = []
    allInfo = []

    fileNormal = config.PATH_RQA + 'normalData.npy'
    fileApnea = config.PATH_RQA + 'apneaData.npy'
    dataNormal = []
    dataApnea = []

    for i_data, data in enumerate(rriData):
        print('record ', i_data, ' len of item ', i_data, len(data), len(label[i_data]))
        content = []
        if len(data) > winSize:
            for start in tqdm(range(len(data) - winSize)):
                end = start + winSize
                timeSeries = data[start:end]
                thisLabel = myUtil.getLabel(label[i_data][start:end])
                rqa = getRQA(timeSeries, dim=dim, tau=tau, epsilon=e, lambd=myLambda, distNorm=disNorm)
                # print(binaryMatrix)
                info = [start, end, thisLabel]

                content.append(info + rqa)

                allRqa.append(rqa)
                allInfo.append([i_data] + info)

                if thisLabel == config.NORMAL_LABEL:
                    dataNormal.append(rqa)
                else:
                    dataApnea.append(rqa)


        else:
            print(" len of data < winSize({})".format(winSize))
        if len(content) > 0:
            outputFileName = config.PATH_RQA + str(i_data) + '.npy'
            print('save file: ', outputFileName)
            np.save(outputFileName, content)

    np.save(fileData, allRqa)
    np.save(fileInfo, allInfo)
    # np.save(fileNormal, dataNormal)
    # np.save(fileApnea, dataApnea)
