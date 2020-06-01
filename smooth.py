import pandas as pd
import numpy as np
import os


def smooth(data_path, weight=0.96):
    f = open(data_path)
    data_str = f.readlines()
    data = []
    for s in data_str:
        data.append(float(s))
    data = np.array(data)
    # data = pd.read_csv(filepath_or_buffer=csv_path,header=0,names=['Step','Value'],dtype={'Step':np.int,'Value':np.float})
    # scalar = data['Value'].values
    scalar = data
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    print(smoothed)


if __name__ == '__main__':
    smooth('mig.txt')
