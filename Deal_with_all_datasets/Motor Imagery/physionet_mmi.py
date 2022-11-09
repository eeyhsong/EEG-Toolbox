from mne.io import RawArray, concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne
import json
import numpy as np
from scipy import io

for sub_num in range(109):
    sub_num = 103
    sub_num = str(sub_num + 1).zfill(3)
    data = []
    label = []
    for run_num in [4, 8, 12]:  # imagine opening and closing left or right fist
        run_num = str(run_num).zfill(2)
        data_path = './eeg-motor-movementimagery-dataset-1.0.0/files/S' + sub_num + '/S' + sub_num + 'R'\
                    + run_num + '.edf'
        raw = mne.io.read_raw_edf(data_path, preload=False)
        events_from_annot, event_dict = mne.events_from_annotations(raw)
        eeg = raw.to_data_frame()
        eeg = np.array(eeg)
        # eeg = eeg[:, [34, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 50, 51, 52, 58]]

        for sam_num in range(np.shape(events_from_annot)[0]):
            begin = events_from_annot[sam_num, 0]
            tmp = eeg[begin:begin+640, :]
            if events_from_annot[sam_num, 2] != 1:
                data.append(tmp)
                label.append(events_from_annot[sam_num, 2] - 2)


    for run_num in [6, 10, 14]:
        run_num = str(run_num).zfill(2)
        data_path = './eeg-motor-movementimagery-dataset-1.0.0/files/S' + sub_num + '/S' + sub_num + 'R' \
                    + run_num + '.edf'
        raw = mne.io.read_raw_edf(data_path, preload=False)
        events_from_annot, event_dict = mne.events_from_annotations(raw)
        eeg = raw.to_data_frame()
        eeg = np.array(eeg)
        # eeg = eeg[:, [34, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 50, 51, 52, 58]]

        for sam_num in range(np.shape(events_from_annot)[0]):
            if events_from_annot[sam_num, 2] == 3:
                begin = events_from_annot[sam_num, 0]
                tmp = eeg[begin:begin + 640, :]
                data.append(tmp)
                label.append(events_from_annot[sam_num, 2] - 1)
            elif events_from_annot[sam_num, 2] == 2:
                begin = events_from_annot[sam_num, 0]
                tmp = eeg[begin:begin + 640, :]
                data.append(tmp)
                label.append(events_from_annot[sam_num, 2] + 1)

    data = np.array(data)
    data = data[:, :, 1:65]
    io.savemat('./eeg-motor-movementimagery-dataset-1.0.0/MMI_four_class_mat/S' + str(sub_num) + '.mat', {'data': data, 'label': label})

    # np.savez('D:/Lab/MI/eeg-motor-movementimagery-dataset-1.0.0/MMI_five_class/S'+sub_num, data, label)

    # print(1)



