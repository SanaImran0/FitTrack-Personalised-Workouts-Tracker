from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd
from pydantic import BaseModel


class Workout(BaseModel):
    acc_x: float
    acc_y: float
    acc_z: float
    gyr_x: float
    gyr_y: float
    gyr_z: float
