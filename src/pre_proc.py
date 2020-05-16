import wfdb
import matplotlib.pyplot as plt
import numpy as np
# from hrv.filters import quotient, moving_median
from scipy import interpolate
from tqdm import tqdm
import os

FS = 100.0


# From https://github.com/rhenanbartels/hrv/blob/develop/hrv/classical.py
def create_time_info(rri):
    rri_time = np.cumsum(rri) / 1000.0  # make it seconds
    return rri_time - rri_time[0]  # force it to start at zero


def create_interp_time(rri, fs):
    time_rri = create_time_info(rri)
    return np.arange(0, time_rri[-1], 1 / float(fs))


def interp_cubic_spline(rri, fs):
    time_rri = create_time_info(rri)
    time_rri_interp = create_interp_time(rri, fs)
    tck = interpolate.splrep(time_rri, rri, s=0)
    rri_interp = interpolate.splev(time_rri_interp, tck, der=0)
    return time_rri_interp, rri_interp


def interp_cubic_spline_qrs(qrs_index, qrs_amp, fs):
    time_qrs = qrs_index / float(FS)
    time_qrs = time_qrs - time_qrs[0]
    time_qrs_interp = np.arange(0, time_qrs[-1], 1 / float(fs))
    tck = interpolate.splrep(time_qrs, qrs_amp, s=0)
    qrs_interp = interpolate.splev(time_qrs_interp, tck, der=0)
    return time_qrs_interp, qrs_interp

outPath = '../output/'
dataPath = '../res/origin/'
trainDataName = ['a01', 'a02', 'a03', 'a04', 'a05',
                   'a06', 'a07', 'a08', 'a09', 'a10',
                   'a11', 'a12', 'a13', 'a14', 'a15',
                   'a16', 'a17', 'a18', 'a19',
                   'b01', 'b02', 'b03', 'b04',
                   'c01', 'c02', 'c03', 'c04', 'c05',
                   'c06', 'c07', 'c08', 'c09',

                   'a20', 'b05', 'c10'
                 ]
# test_data_name = ['a20', 'b05', 'c10']
# age = [51, 38, 54, 52, 58,
#        63, 44, 51, 52, 58,
#        58, 52, 51, 51, 60,
#        44, 40, 52, 55, 58,
#        44, 53, 53, 42, 52,
#        31, 37, 39, 41, 28,
#        28, 30, 42, 37, 27]
# sex = [1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1,
#        0, 1, 1, 1, 1,
#        1, 1, 1, 0, 0,
#        0, 0, 1, 1, 1]


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
FS_INTP = 4
MAX_HR = 300.0
MIN_HR = 20.0
MIN_RRI = 1.0 / (MAX_HR / 60.0) * 1000
MAX_RRI = 1.0 / (MIN_HR / 60.0) * 1000
train_input_array = []
train_label_array = []

for dataIndex, dataName in enumerate(trainDataName):

    print(dataName)
    numberOfLabel = len(wfdb.rdann(os.path.join(dataPath, trainDataName[dataIndex]), 'apn').symbol)
    signals, fields = wfdb.rdsamp(os.path.join(dataPath, trainDataName[dataIndex]))
    for index in tqdm(range(1, numberOfLabel)):
        sampFrom = index * 60 * FS  # 60 seconds
        sampTo = sampFrom + 60 * FS  # 60 seconds

        # array dữ liệu sạch
        # from -> to: 80 seconds
        qrsAnn = wfdb.rdann(dataPath + trainDataName[dataIndex], 'qrs', sampfrom=sampFrom - (MARGIN * 100),
                            sampto=sampTo + (MARGIN * 100)).sample
        # qrs_ann là thời gian
        # plt.plot(qrs_ann, color='red')
        # plt.show()
        # lấy label của dữ liệu apnea
        # from -> to: 6000đv - 1đv
        apnAnn = wfdb.rdann(dataPath + trainDataName[dataIndex], 'apn', sampfrom=sampFrom,
                            sampto=sampTo - 1).symbol
        # lấy tín hiệu ecg cao nhất
        # qrs_amp = get_qrs_amp(signals, qrs_ann)

        # diff: out[i] = a[i+1] - a[i] -> Tinh khoang RR, rri la chuoi cac gia tri RR
        rri = np.diff(qrsAnn)
        rriByMs = rri.astype('float') / FS * 1000.0

        if apnAnn[0] == 'N': # Normal
            label = 0.0
        elif apnAnn[0] == 'A': # Apnea
            label = 1.0
        else:
            label = 2.0

        if (index in [1, 2, 3, 4, 5, numberOfLabel-1]):
            print(dataIndex, rriByMs)
        train_input_array.append([rriByMs, dataIndex, dataName])
        train_label_array.append(label)
# print("all len: ", len(train_input_array))
np.save(outPath + 'my_train_input.npy', train_input_array)
np.save(outPath + 'my_train_label.npy', train_label_array)