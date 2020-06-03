import numpy as np
from src import config as config
from src import MyUtil as myUtil
from tqdm import tqdm
from src import RecurrentPlot as rp
from src import RecurrenceQuantificationAnalysis as rqa

import time
from threading import Thread
from multiprocessing import Process

for i in range(0, 45, 5):
    print('r: ', i)
a = 100


def testThread(name='T1', sleepTime=1):
    for i in range(50):
        print('threadName: {}, sleepTime: {}'.format(name, a))
        time.sleep(sleepTime)


import datetime

currentDT = datetime.datetime.now()
print(str(currentDT))

if __name__ == '__main__':
    t1 = Process(target=testThread, args=('T1', 1))
    t2 = Process(target=testThread, args=('T2', 2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
