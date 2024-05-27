import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# from sklearn.decomposition import PCA
# from scipy.signal import butter, lfilter, filtfilt
# import copy
# import scipy.stats as stats

# # --------------------------------------------------------------
# # DataTransformation
# # --------------------------------------------------------------

# class LowPassFilter:
#     def low_pass_filter(
#         self,
#         data_table,
#         col,
#         sampling_frequency,
#         cutoff_frequency,
#         order=5,
#         phase_shift=True,
#     ):
#         # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
#         # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
#         nyq = 0.5 * sampling_frequency
#         cut = cutoff_frequency / nyq

#         b, a = butter(order, cut, btype="low", output="ba", analog=False)
#         if phase_shift:
#             data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
#         else:
#             data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
#         return data_table


# # Class for Principal Component Analysis. We can only apply this when we do not have missing values (i.e. NaN).
# # For this we have to impute these first, be aware of this.
# class PrincipalComponentAnalysis:

#     pca = []

#     def __init__(self):
#         self.pca = []

#     def normalize_dataset(self, data_table, columns):
#         dt_norm = copy.deepcopy(data_table)
#         for col in columns:
#             dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
#                 data_table[col].max()
#                 - data_table[col].min()
#                 # data_table[col].std()
#             )
#         return dt_norm

#     # Perform the PCA on the selected columns and return the explained variance.
#     def determine_pc_explained_variance(self, data_table, cols):

#         # Normalize the data first.
#         dt_norm = self.normalize_dataset(data_table, cols)

#         # perform the PCA.
#         self.pca = PCA(n_components=len(cols))
#         self.pca.fit(dt_norm[cols])
#         # And return the explained variances.
#         return self.pca.explained_variance_ratio_

#     # Apply a PCA given the number of components we have selected.
#     # We add new pca columns.
#     def apply_pca(self, data_table, cols, number_comp):

#         # Normalize the data first.
#         dt_norm = self.normalize_dataset(data_table, cols)

#         # perform the PCA.
#         self.pca = PCA(n_components=number_comp)
#         self.pca.fit(dt_norm[cols])

#         # Transform our old values.
#         new_values = self.pca.transform(dt_norm[cols])

#         # And add the new ones:
#         for comp in range(0, number_comp):
#             data_table["pca_" + str(comp + 1)] = new_values[:, comp]

#         return data_table
    
    

# # --------------------------------------------------------------
# # TemporalAbstraction
# # --------------------------------------------------------------

# class NumericalAbstraction:

#     # This function aggregates a list of values using the specified aggregation
#     # function (which can be 'mean', 'max', 'min', 'median', 'std')
#     def aggregate_value(self, aggregation_function):
#         # Compute the values and return the result.
#         if aggregation_function == "mean":
#             return np.mean
#         elif aggregation_function == "max":
#             return np.max
#         elif aggregation_function == "min":
#             return np.min
#         elif aggregation_function == "median":
#             return np.median
#         elif aggregation_function == "std":
#             return np.std
#         else:
#             return np.nan

#     # Abstract numerical columns specified given a window size (i.e. the number of time points from
#     # the past considered) and an aggregation function.
#     def abstract_numerical(self, data_table, cols, window_size, aggregation_function):

#         # Create new columns for the temporal data, pass over the dataset and compute values
#         for col in cols:
#             data_table[
#                 col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
#             ] = (
#                 data_table[col]
#                 .rolling(window_size)
#                 .apply(self.aggregate_value(aggregation_function))
#             )

#         return data_table

# # --------------------------------------------------------------
# # FrequencyAbstraction
# # --------------------------------------------------------------

# class FourierTransformation:

#     # Find the amplitudes of the different frequencies using a fast fourier transformation. Here,
#     # the sampling rate expresses the number of samples per second (i.e. Frequency is Hertz of the dataset).
#     def find_fft_transformation(self, data, sampling_rate):
#         # Create the transformation, this includes the amplitudes of both the real
#         # and imaginary part.
#         transformation = np.fft.rfft(data, len(data))
#         return transformation.real, transformation.imag

#     # Get frequencies over a certain window.
#     def abstract_frequency(self, data_table, cols, window_size, sampling_rate):

#         # Create new columns for the frequency data.
#         freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

#         for col in cols:
#             data_table[col + "_max_freq"] = np.nan
#             data_table[col + "_freq_weighted"] = np.nan
#             data_table[col + "_pse"] = np.nan
#             for freq in freqs:
#                 data_table[
#                     col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
#                 ] = np.nan

#         # Pass over the dataset (we cannot compute it when we do not have enough history)
#         # and compute the values.
#         for i in range(window_size, len(data_table.index)):
#             for col in cols:
#                 real_ampl, imag_ampl = self.find_fft_transformation(
#                     data_table[col].iloc[
#                         i - window_size : min(i + 1, len(data_table.index))
#                     ],
#                     sampling_rate,
#                 )
#                 # We only look at the real part in this implementation.
#                 for j in range(0, len(freqs)):
#                     data_table.loc[
#                         i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
#                     ] = real_ampl[j]
#                 # And select the dominant frequency. We only consider the positive frequencies for now.

#                 data_table.loc[i, col + "_max_freq"] = freqs[
#                     np.argmax(real_ampl[0 : len(real_ampl)])
#                 ]
#                 data_table.loc[i, col + "_freq_weighted"] = float(
#                     np.sum(freqs * real_ampl)
#                 ) / np.sum(real_ampl)
#                 PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
#                 PSD_pdf = np.divide(PSD, np.sum(PSD))
#                 data_table.loc[i, col + "_pse"] = -np.sum(np.log(PSD_pdf) * PSD_pdf)

#         return data_table

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
df.info()

# Taking instances out of the data for runtime testing
test1_df = df[0:2].drop(["participant", "label",	"category",	"set"], axis=1)
test1_label = "bench"
test1_df.to_csv("../../data/external/test1.csv")

test2_df = df[9000:9002].drop(["participant", "label",	"category",	"set"], axis=1)
test2_label = "row"
test2_df.to_csv("../../data/external/test2.csv")

test3_df = df[4567:4569].drop(["participant", "label",	"category",	"set"], axis=1)
test3_label = "dead"
test3_df.to_csv("../../data/external/test3.csv")


# The test csv file for run-time testing
test = pd.concat([test1_df, test2_df, test3_df])
test.to_csv("../../data/external/test.csv")


# Removing test instances from dataset
df = df[2:]
df = df.drop(["2019-01-20 17:33:26.600", "2019-01-20 17:33:26.800"], axis=0)
df = df.drop(["2019-01-15 19:26:50.800", "2019-01-15 19:26:51.000"], axis=0)


predictor_columns = list(df.columns[:6])
set64 = df.head(85)
df[df["set"] == 64].count()
df.head(87)
# Plot settings
# `plt.style.use("fivethirtyeight")` sets the style of the plots to the "fivethirtyeight" style, which
# is a popular style for data visualization.
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()
subset = df[df["set"] == 35]["gyr_y"].plot()

for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info()    

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    
    duration = stop - start
    
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col+"_lowpass"]
    del df_lowpass[col+"_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

PCAa = PrincipalComponentAnalysis()
pca_values = PCAa.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1, len(predictor_columns)+ 1), pca_values)
plt.xlabel("Principle Component Number")
plt.ylabel("Explained Variance")
plt.show()

df_pca = PCAa.apply_pca(df_pca, predictor_columns, 3)
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] **2 + df_squared["acc_y"] **2 + df_squared["acc_z"] **2 
gyr_r = df_squared["gyr_x"] **2 + df_squared["gyr_y"] **2 + df_squared["gyr_z"] **2 

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000/200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)    

df_temporal = pd.concat(df_temporal_list)

df_temporal.info()
    
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

sub = df_temporal[df_temporal["set"] == 15]

fs = int(1000/200)
ws = int(2800/200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq.columns

# Visualize results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        'acc_y_max_freq', 
        'acc_y_freq_weighted',
        'acc_y_pse',
        'acc_y_freq_1.429_Hz_ws_14',
        'acc_y_freq_2.5_Hz_ws_14'
    ]
].plot()

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformation to set {s}: ")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index('epoch (ms)', drop=True)
df_freq.info()

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(10,10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of Squared Distances")
plt.show()    

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot Clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
    
ax.set_xlabel("X-axis")    
ax.set_ylabel("Y-axis")   
ax.set_zlabel("Z-axis")   
plt.legend()
plt.show()

# Plot Accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
    
ax.set_xlabel("X-axis")    
ax.set_ylabel("Y-axis")   
ax.set_zlabel("Z-axis")   
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
df_cluster.to_csv("../../data/interim/03_data_features.csv")

