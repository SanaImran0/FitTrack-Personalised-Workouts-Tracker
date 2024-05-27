import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from pydantic import BaseModel
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import scipy.stats as stats
from pydantic import BaseModel


class Workout(BaseModel):
    acc_x: float
    acc_y: float
    acc_z: float
    gyr_x: float
    gyr_y: float
    gyr_z: float


# class LowPassFilter:
#     def low_pass_filter(
#         self,
#         data,
#         key,
#         sampling_frequency,
#         cutoff_frequency,
#         order=4,
#         phase_shift=True,
#     ):
#         nyq = 0.5 * sampling_frequency
#         cut = cutoff_frequency / nyq

#         b, a = butter(order, cut, btype="low", output="ba", analog=False)
#         if phase_shift:
#             data[key + "_lowpass"] = filtfilt(b, a, data[key])
#         else:
#             data[key + "_lowpass"] = lfilter(b, a, data[key])
#         return data


class PrincipalComponentAnalysis:
    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data, keys):
        dt_norm = copy.deepcopy(data)
        for key in keys:
            dt_norm[key] = (data[key] - data[key].mean()) / (
                data[key].max()
                - data[key].min()
                # data[key].std()
            )
        return dt_norm

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data, keys):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data, keys)

        # perform the PCA.
        self.pca = PCA(n_components=len(keys))
        self.pca.fit(dt_norm[keys])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    # Apply a PCA given the number of components we have selected.
    # We add new pca columns.
    def apply_pca(self, data, keys, number_comp):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data, keys)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[keys])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[keys])

        # And add the new ones:
        for comp in range(0, number_comp):
            data["pca_" + str(comp + 1)] = new_values[:, comp]

        return data


class NumericalAbstraction:

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std')
    def aggregate_value(self, aggregation_function):
        # Compute the values and return the result.
        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            return np.nan

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    def abstract_numerical(self, data, keys, window_size, aggregation_function):

        # Create new columns for the temporal data, pass over the dataset and compute values
        for key in keys:
            data[
                key + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            ] = (
                data[key]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function))
            )

        return data
    

class FourierTransformation:

    # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
    # the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
    def find_fft_transformation(self, data, sampling_rate):
        # Create the transformation, this includes the amplitudes of both the real
        # and imaginary part.
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    # Get frequencies over a certain window.
    def abstract_frequency(self, data, keys, window_size, sampling_rate):

        # Create new columns for the frequency data.
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for key in keys:
            data[key + "_max_freq"] = np.nan
            data[key + "_freq_weighted"] = np.nan
            data[key + "_pse"] = np.nan
            for freq in freqs:
                data[
                    key + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        # Pass over the dataset (we cannot compute it when we do not have enough history)
        # and compute the values.
        for i in range(window_size, len(data.index)):
            for key in keys:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data[key].iloc[
                        i - window_size : min(i + 1, len(data.index))
                    ],
                    sampling_rate,
                )
                # We only look at the real part in this implementation.
                for j in range(0, len(freqs)):
                    data.loc[
                        i, key + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                    ] = real_ampl[j]
                # And select the dominant frequency. We only consider the positive frequencies for now.

                data.loc[i, key + "_max_freq"] = freqs[
                    np.argmax(real_ampl[0 : len(real_ampl)])
                ]
                data.loc[i, key + "_freq_weighted"] = float(
                    np.sum(freqs * real_ampl)
                ) / np.sum(real_ampl)
                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))
                data.loc[i, key + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

        return data
    
    

app = FastAPI()
model_file_path = '/Users/macbookpro/Desktop/FYP/ML Fitness Tracker/src/random_forest_model.pkl'
model = pickle.load(open(model_file_path, 'rb'))


@app.get('/')
def index():
    return {'message': 'Hello!'}


@app.get('/{name')
def get_name(name: str):
    return {'Welcome back ': f'{name}'}


data = {
    'acc_x': np.array([0.0135, -0.0015]),
    'acc_y': np.array([0.977, 0.9704999999999999]),
    'acc_z': np.array([-0.071, -0.0795]),
    'gyr_x': np.array([-1.8904, -1.6826]),
    'gyr_y': np.array([2.4391999999999996, -0.8904]),
    'gyr_z': np.array([0.9388000000000002, 2.1708])    
}

data.keys()
data.values()

@app.post('/predict')
def predict_workout(data: Workout):
    data = dict(data)
    acc_x = data['acc_x']
    acc_y = data['acc_y']
    acc_z = data['acc_z']
    gyr_x = data['gyr_x']
    gyr_y = data['gyr_y']
    gyr_z = data['gyr_z']
    
    keys = ('acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z')

    # LowPass = LowPassFilter()

    # fs = 0.1
    # cutoff = 0.1
    # for key in data:
    #     data = LowPass.low_pass_filter(data, key, fs, cutoff, order=4)
    #     data[key] = data[key + '_lowpass']
    #     del data[key + '_lowpass']
        
    PCAa = PrincipalComponentAnalysis()
    #pca_values = PCAa.determine_pc_explained_variance(data, keys)
    data = PCAa.apply_pca(data, data.keys(), 3)
    
    acc_r = data["acc_x"] **2 + data["acc_y"] **2 + data["acc_z"] **2 
    gyr_r = data["gyr_x"] **2 + data["gyr_y"] **2 + data["gyr_z"] **2 

    data["acc_r"] = np.sqrt(acc_r)
    data["gyr_r"] = np.sqrt(gyr_r)
    
    NumAbs = NumericalAbstraction()
    keys = data.keys() + ["acc_r", "gyr_r"]
    
    ws = int(1000/200)
    for key in keys:
        data = NumAbs.abstract_numerical(data, [key], ws, "mean")
        data = NumAbs.abstract_numerical(data, [key], ws, "std")
        
    FreqAbs = FourierTransformation()
    
    fs = 1
    ws = 1
    
    data = FreqAbs.abstract_frequency(data, keys, ws, fs)
    
    prediction = model.predict([data.values()])
    if(prediction[0] == "bench"):
        prediction = "You did bench workout!"
    elif(prediction[0] == "dead"):
        prediction = "You did deadlifts!" 
    elif(prediction[0] == "ohp"):
        prediction = "You did over-head-press!"  
    elif(prediction[0] == "squat"):
        prediction = "You did squats!"
    elif(prediction[0] == "row"):
        prediction = "You did row workout!"
    else:
        prediction = "You were resting!" 
        
    return { 'Prediction: ': prediction}

predict_workout(data)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    # uvicorn app:app --reload