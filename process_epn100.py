import gc
from os.path import join
import numpy as np 
import copy

import libemg
from libemg.datasets import get_dataset_list
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics

from utils import *


# ======== DATA ========
from EPN100 import EMGEPN100
dataset = EMGEPN100()
data = dataset.prepare_data(split=True, segment=True, relabel_seg=0)

train_data = data["Train"].isolate_data("devices", [1], fast=True)      # gForce is 500Hz
test_data = data["Test"].isolate_data("devices", [1], fast=True)

val_data = test_data.isolate_data("subjects", list(range(43, VAL_CUTOFF)), fast=True)
test_data = test_data.isolate_data("subjects", list(range(VAL_CUTOFF, 85)), fast=True)

np.save(join(PATH, 'train_data'), train_data)
np.save(join(PATH, 'val_data'), val_data)
np.save(join(PATH, 'test_data'), test_data)

train_windows, train_meta = train_data.parse_windows(SEQ, INC)
val_windows, val_meta = val_data.parse_windows(SEQ, INC)
test_windows, test_meta = test_data.parse_windows(SEQ, INC)

np.save(join(PATH, 'train_windows'), train_windows.astype(DTYPE))
np.save(join(PATH, 'train_meta'), train_meta)

np.save(join(PATH, 'val_windows'), val_windows.astype(DTYPE))
np.save(join(PATH, 'val_meta'), val_meta)

np.save(join(PATH, 'test_windows'), test_windows.astype(DTYPE))
np.save(join(PATH, 'test_meta'), test_meta)


print('DONE.')