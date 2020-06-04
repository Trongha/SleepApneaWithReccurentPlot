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

for i in range(0, 10):
    startTime = time.time()
    print(str(startTime))
    print('=======> Load cluster ', i, ' . . . ')

    recordNames = config.NAME_OF_RECORD[1:5]
    listRp, listLabel, _ = myUtil.loadRpByCluster(10, 2, listRecordNames=recordNames, type='train')
    print(listRp.shape)
    print(listLabel.shape)
    endTime = time.time()
    print('duration: ', endTime-startTime)



