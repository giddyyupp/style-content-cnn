import config as cfg
import os
from os import listdir
import numpy as np
from os.path import isfile, join


def prep_data(main_path):
    files_all = []
    labels_all = []
    labels_temp = []
    # get all folders for each class
    test_f = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
    # test_f = [test_folders[0] for test_folders in os.walk(main_path)][1:]
    # get files for each folder
    for folder in test_f:
        index_ = cfg.CLASS_LIST.index(folder)
        test_ff = [d for d in os.listdir(os.path.join(main_path, folder)) if os.path.isdir(os.path.join(main_path, folder, d))]
        if len(test_ff):
            for fff in test_ff:
                test_files = [files_all.append(os.path.join(main_path, folder, fff, f)) for f in listdir(os.path.join(main_path, folder, fff))
                              if isfile(join(main_path, folder, fff, f))]
                # files_all.append(test_files)
                labels_temp.append(np.zeros((len(test_files),), dtype=int)[:] + index_)
        else:
            test_files = [files_all.append(os.path.join(main_path, folder, f)) for f in listdir(main_path + folder) if isfile(join(main_path + folder, f))]
            # files_all.append(test_files)
            labels_temp.append(np.zeros((len(test_files),),dtype=int)[:] + index_)

    for lst in labels_temp:
        for i in lst:
            labels_all.append(i)

    return files_all, labels_all
