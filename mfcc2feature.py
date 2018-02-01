import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import normalize

rate = 100


def extractSpeechData(path):
    data = []
    data_point = []
    file_object = open(path)
    if file_object:
        i = 0
        num_string = str(i) + ":"
        for line in file_object:
            data_line = line.split()
            if num_string == data_line[0] or "END" in data_line:
                del data_point[0]
                data.append(data_point)
                data_point = []
                data_point.extend(data_line)
                i += 1
                num_string = str(i) + ":"
            else:
                data_point.extend(data_line)
        file_object.close()
    else:
        print("Unable to open Speech Data File")
    del data[0]

    converted_file = []
    for data_line in data:
        converted_data = []
        for data_point in data_line:
            converted_data.append(float(data_point))
        converted_file.append(converted_data)
    data = converted_file

    check = 0
    frame = 0
    formed_data = []

    for data_feature in data:
        formed_data.append(frame)
        formed_data.extend(data_feature)
        if check == 0:
            array_data = np.asarray(formed_data)
            check = 1
        else:
            b = np.asarray(formed_data)
            array_data = np.vstack((array_data, b))
        frame += 1
        formed_data = []
    return array_data


def upSampling(feature, rate):
    period = len(feature)
    period_index = pd.date_range(start='1/1/2000', periods=period, freq='s')
    pd_x = pd.DataFrame(feature, index=period_index)

    pd_x = pd_x.resample(str(int(rate)) + 's')
    pd_x = pd_x.sum()
    upsampled_x = pd_x.interpolate(method='cubic')
    upsampled_x = upsampled_x.as_matrix().reshape([np.shape(upsampled_x)[0], ])

    return upsampled_x


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        a = np.asarray(extractSpeechData(arg))
        b = np.empty([1, ])
        for n in range(1, len(a[0])):
            c = np.reshape(normalize(upSampling(a[:, n], rate)), (30,))
            b = np.column_stack((b, c))
        np.save(arg, b[:, 1:])





