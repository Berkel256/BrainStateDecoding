### Copyright for the code by Alexandra Berkel, 2018 ###
### This code was written for the Brain State Decoding Lab Seminar in the University of Freiburg.
### It reproduces the experiment from "Multiscale temporal neural dynamics predict performance in a complex sensorimotor task" by Samek et al. (2016).
### The EEG data is downloaded from http://bbci.de/competition/iv provided thanks to "The noninvasive berlin brain-computer interface: Fast acquisition of effective performance in untrained subjects" by Blankertz et al. (2007)
### This code is able to read the EEG file of one participant. It computes the Hurst exponent for six CSP components in one frequency range. It also provides plots from the computational steps.

# Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy
import sklearn
from sklearn import linear_model
from scipy.signal import hilbert, butter, lfilter


##### Read BCI data from Blankertz #####

# read eeg file from Blankertz BCI data
# file: timestamps x channels
# return eeg: channels x timestamps
def read_eeg(file):
    with open(file) as eeg_file:
        lines = eeg_file.readlines()
        t = len(lines)
        c = len(lines[0].split("\t"))
        eeg = numpy.zeros((c, t), dtype=float)

        for t_nr, line in enumerate(lines):
            values = []
            for c_nr, value in enumerate(line.split("\t")):
                value = float(value) * 0.1
                eeg[c_nr][t_nr] = value

        return eeg
        
# get timestamps as seconds starting from 0 for the length of the data
def get_timestamps(length, samplingrate):
    ts = [i/float(samplingrate) for i in range(length)]
    return ts

fixation_id = 0
cue_id = 1
rest_id = 2
pause_id = 3
def get_timestamps_information(max_timestamp, samplingrate, fixation_sec, cue_sec, rest_sec, pause_sec, pause_trials):
    trial_ids = [fixation_id for i in range(fixation_sec * samplingrate)]
    trial_ids.extend([cue_id for i in range(cue_sec * samplingrate)])
    trial_ids.extend([rest_id for i in range(rest_sec * samplingrate)])
    pause_ids = [pause_id for i in  range(pause_sec * samplingrate)]
    information = []
    t = 0
    trial = 0
    while t < max_timestamp:
        if trial % pause_trials == 0 and trial > 0:
            information.extend(pause_ids)
            t += len(pause_ids)
        information.extend(trial_ids)
        t += len(trial_ids)
        trial += 1
        
    return information[:max_timestamp]


##### Preprocessing #####

# return CSP filters x channels
def get_CSP_filters(data, count):
    # order eigenvectors of covariance according to descending eigenvalues
    covariance = numpy.cov(data)
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)    
     
    # compute median of variances
    variances = []
    for csp_filter in eigenvectors:
        med_var = numpy.median(numpy.var(csp_filter.dot(data)))
        variances.append(med_var)
    scores = variances / sum(variances)
    
    # select lowest and highest scores
    scores_sorting = numpy.argsort(scores)
    elements = int(numpy.ceil(float(count) / 2.0)) 
    lowest_idx = scores_sorting[range(0, elements)]
    highest_idx = scores_sorting[range(len(scores)-elements, len(scores))]
    relevant_idx = numpy.append(lowest_idx, highest_idx)

    # sort filters according to discriminativity
    CSP_filters = []
    relevant_scores = [max(s, 1-s) for s in scores[relevant_idx]]
    disc_sorting = numpy.argsort(relevant_scores)[::-1]
    relevant_idx = relevant_idx[disc_sorting]
    CSP_filters = eigenvectors[relevant_idx]

    return CSP_filters

# return CSP pattern x timestamps
def apply_CSP_filters(data, CSP_filters):
    CSP_filtered = numpy.dot(CSP_filters, data)
    return CSP_filtered

# returns outliers as Boolean array, where True indicates an outlier in the data with the same index
def find_outliers(data):
    count = len(data)
    outliers = [False for i in range(count)]
    find_outliers = True
    
    # recurse while there are outliers
    while(find_outliers):
        outlierfree_data = [v for idx, v in enumerate(data) if outliers[idx]==False]
        mean = numpy.mean(outlierfree_data)
        standard_deviation = numpy.std(outlierfree_data)
        find_outliers = False 
        
        for i, is_outlier in enumerate(outliers):
            if is_outlier:
                continue
            val = data[i]
            if abs(val - mean) > 4 * standard_deviation:
                outliers[i] = True
                find_outliers = True
                
    return outliers 
    
# get narrowband with 5th order Butterworth filter
def get_narrowband(data, lowcut, highcut):
    order = 5
    nyq = 0.5 * samplingrate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    narrow = lfilter(b, a, data)
    return narrow

# get envelope with Hilbert transform
def get_envelope(data):
    return numpy.abs(hilbert(data))

# get predictions for timestamps for linearly fit data
def fit_linear(x, y):
    reg = linear_model.LinearRegression()
    x = numpy.array(x).reshape(-1, 1)
    model = reg.fit(x, y)
    fit = model.predict(x)
    slope = model.coef_[0]
    return fit, slope


##### DFA Steps #####

# get cumulative sum for all timepoints
def integrate(data):
    cumulative_sums = []
    cumulative_sum = 0
    for i in range(len(data)):
        cumulative_sum += data[i]
        cumulative_sums.append(cumulative_sum)
    return cumulative_sums
    
# split data into windows of given length, remainder is omitted
def get_windows(data, timestamps, window_length):
    window_count = len(data) / window_length
    reduced = window_count * window_length
    windows_timestamps = numpy.array(timestamps[:reduced]).reshape(window_count, window_length)
    windows = numpy.array(data[:reduced]).reshape(window_count, window_length)
    return windows, windows_timestamps

# reject windows containing outliers
def reject_outlier_windows(windows, windows_timestamps, outliers):
    # bring outliers in shape of windows
    window_count = len(windows)
    window_length = len(windows[0])
    reduced = window_count * window_length
    windows_outliers = numpy.array(outliers[:reduced]).reshape(window_count, window_length)
    
    # select outlierfree windows
    outlierfree_windows = []
    for j in range(window_count):
        is_outlier_window = numpy.any(windows_outliers[j])
        if not is_outlier_window:
            outlierfree_windows.append(windows[j])
    
    # get continuous timestamps
    outlierfree_timestamps = windows_timestamps[:len(outlierfree_windows)]
    
    return outlierfree_windows, outlierfree_timestamps   
    
# detrending with linear fit
def detrending(data, timestamps):
    fit = fit_linear(timestamps, data)[0]
    detrended = data - fit
    return detrended

# sum variances of all windows
def get_variances_sum(windows):
    variance_sum = 0
    for win in windows:
        variance = win.transpose().dot(win)
        variance_sum += variance
    return variance_sum


##### Hurst exponent estimation with DFA #####

# calculate DFA coefficient for given window length
def get_DFA_coefficient(data, timestamps, outliers, window_length,):
    cum_sum = integrate(data)
    windows, windows_timestamps = get_windows(cum_sum, timestamps, window_length)
    n = len(windows[0]) # window length
	N = len(data)
    windows, windows_timestamps = reject_outlier_windows(windows, windows_timestamps, outliers)
    if len(windows) < 10:
        raise Exception("Not enough windows!") 
        
    detrended_windows = []
    for window, window_timestamps in zip(windows, windows_timestamps):
        detrended = detrending(window, window_timestamps)
        detrended_windows.append(detrended)
    DFA_coeff = get_variances_sum(detrended_windows)
    norm = (N/n) * n
    return DFA_coeff/norm

def get_window_range(datapoints, step_size, log_scale=True):
    max_window_length = datapoints
    if log_scale:
        max_window_length = numpy.log2(datapoints)
    window_lengths = []
    window_length = step_size
    
    while (window_length < max_window_length):        
        window_lengths.append(window_length)
        window_length += step_size
        
    return window_lengths

def get_window_coefficients(data, timestamps, outliers, window_range, log_scale=True):
    coefficients = []
    while (len(coefficients) < len(window_range)):
        window_length = window_range[len(coefficients)]
        if log_scale:
            window_length = int(2 ** window_length)
        try:
            coefficient = get_DFA_coefficient(data, timestamps, outliers, window_length)
            if log_scale:
                if coefficient > 0:
                    coefficient = numpy.log(coefficient)
                else: 
                    raise Exception("coefficient: ", coefficient)
            coefficients.append(coefficient)
        except Exception:
            window_range = numpy.delete(window_range, len(coefficients))
    
    return window_range, coefficients

# make linear fit for log(window_length) and log(DFA coefficient)
def get_hurst_exponent(data, timestamps, outliers, window_lengths, log_scale=True):
    # calculate DFA coefficient for all window lengths
    window_lengths, coefficients = get_window_coefficients(data, timestamps, outliers, window_lengths, log_scale)
    
    if not log_scale:
        window_lengths = [numpy.log2(w) for w in window_lengths]
        coefficients = [numpy.log2(c) for c in coefficients]
        
    if (len(window_lengths) > 0):
        # fit log window lengths and log coefficients to get the slope
        hurst_fit, hurst_exponent = fit_linear(window_lengths, coefficients)
        return hurst_exponent



## Read file and apply CSP
eeg_channels = read_eeg("BCICIV_calib_ds1a_cnt.txt")

# variable definitions
channel_count = len(eeg_channels)
samplingrate = 100
fixation_sec = 2
cue_sec = 4 
rest_sec = 2 
trial_sec = fixation_sec + cue_sec + rest_sec
trials = 75
pause_sec = 15 
pause_trials = 20
pauses = trials/pause_trials

start_time = len(eeg_channels[0]) % samplingrate
eeg_channels = eeg_channels[:, start_time:]
end_time = len(eeg_channels[0])

N = len(eeg_channels[0])

# timestamps
timestamps = get_timestamps(end_time, samplingrate)
ts_info = get_timestamps_information(end_time, samplingrate, fixation_sec, cue_sec, rest_sec, pause_sec, pause_trials)
# pause_timestamps = [timestamps[i] for i in range(end_time) if  ts_info[i] == pause_id]



## Example preprocessing
eeg_narrowbands = [get_narrowband(channel, 10, 15) for channel in eeg_channels]
csp_filter_count = 6
csp_filters = get_CSP_filters(eeg_narrowbands, csp_filter_count)
eeg_spatial_filtered = apply_CSP_filters(eeg_narrowbands, csp_filters)



## Example computation of hurst exponents

# prepare range of log window lengths
step_size = 1.0
log_scale = True
window_lengths = get_window_range(len(eeg_spatial_filtered[0]), step_size, log_scale)

# all csp channels
csp_envelope = []
csp_hurst = []
for channel in eeg_spatial_filtered:
    # outliers and pauses
    outliers = find_outliers(channel)
    outliers = [outliers[i] or ts_info[i] == pause_id for i in range(end_time)]
    
    channel_envelope = get_envelope(channel)  
    hurst = get_hurst_exponent(channel_envelope, timestamps, outliers, window_lengths, log_scale)
    
    csp_envelope.append(channel_envelope)
    csp_hurst.append(hurst)
    
print("hurst: ", csp_hurst)


## Helpers for plot example
# get linear fit for all windows and their timestamps
def get_linear_fits(windows, ts):
    fits = []
    for i in range(len(windows)):
        win = windows[i]
        win_ts = ts[i]
        fit = fit_linear(win_ts, win)[0]
        fits.append(fit)
    return fits

# detrend all windows with their fits
def detrend(windows, fits):
    detrended = []
    for win, fit in zip(windows, fits):
        diff = win - fit
        detrended.append(diff)
    return detrended

# first csp channel for plots
csp_component = 0
first_outliers = find_outliers(eeg_spatial_filtered[csp_component])
first_outliers_and_pauses = [first_outliers[i] or ts_info[i] == pause_id for i in range(end_time)]
first_envelope = get_envelope(eeg_spatial_filtered[csp_component])
first_integrated = integrate(first_envelope)

# windowing 
window_count = 25
window_length = (len(timestamps) / window_count / 1000) * 1000
first_windows, first_win_ts = get_windows(first_integrated, timestamps, window_length)
first_outlierfree, first_outlierfree_ts = reject_outlier_windows(first_windows, first_win_ts, first_outliers_and_pauses)
print(len(first_outlierfree))

# DFA coefficients
first_fits = get_linear_fits(first_outlierfree, first_outlierfree_ts)
first_detrended = detrend(first_outlierfree, first_fits)
#first_DFA_coeff = normalized_variances_sum(first_detrended)


##### Example plots preprocessing #####

# EEG channels
plt.plot(timestamps, eeg_channels[0], label="eeg channel", c="black")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# EEG narrowband
plt.plot(timestamps, eeg_narrowbands[0], label="narrowband (10-15 Hz)", c="olive")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# CSP filtered
plt.plot(timestamps, eeg_spatial_filtered[0], label="CSP component", c="green")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()


## Cut outs
trial = 51
start = (trial * 8 + 0 + (trial/20)*15) * samplingrate
stop = ((trial+0) * 8 + 6 + (trial/20)*15) * samplingrate


# EEG channels
for i in range(1, channel_count):
    plt.plot(timestamps[start:stop], eeg_channels[i][start:stop], c="black", alpha=0.02)
plt.plot(timestamps[start:stop], eeg_channels[0][start:stop], 
         label="eeg channel " + str(1) + "/" + str(channel_count), c="black")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

#EEG channels with motor imagery
#for i in range(1, channel_count):
#    plt.plot(timestamps[start:stop], eeg_channels[i][start:stop], c="black", alpha=0.02)
#plt.plot(timestamps[start:stop], eeg_channels[0][start:stop], 
#         label="eeg channel " + str(1) + "/" + str(channel_count), c="black")
#show_label = True
#i = start
#while i < stop:
#    if ts_info[i] == cue_id:
#        x_left = timestamps[i]
#        x_right = x_left
#        is_cue = True
#        while is_cue:
#            i += 1
#            if ts_info[i] == cue_id:
#                is_cue = True
#            else:
#                is_cue = False
#                x_right = timestamps[i]
#                if show_label:
#                    plt.axvspan(xmin = x_left, xmax = x_right, color="yellow", label="motor imagery")
#                    show_label = False
#                else:
#                    plt.axvspan(xmin = x_left, xmax = x_right, color="yellow")
#    i += 1
#plt.xlabel("time [sec]")
#plt.ylabel("power [uV]")
#plt.legend()
#plt.show()

# EEG narrowband cut out
for i in range(1, channel_count):
    plt.plot(timestamps[start:stop], eeg_narrowbands[i][start:stop], c="olive", alpha=0.01)
plt.plot(timestamps[start:stop], eeg_narrowbands[0][start:stop], 
         label="narrowband " + str(1) + "/" + str(channel_count) + " (10-15 Hz)", c="olive")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# CSP filtered cut out
for i in range(csp_filter_count):
    plt.plot(timestamps[start:stop], eeg_spatial_filtered[i][start:stop], c="green", alpha=0.15)
plt.plot(timestamps[start:stop], eeg_spatial_filtered[csp_component][start:stop], 
         label="CSP component " + str(csp_component+1) + "/" + str(csp_filter_count), c="green")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()


# envelope
for i in range(csp_filter_count):
    plt.plot(timestamps[start:stop], csp_envelope[i][start:stop], c="steelblue", alpha=0.15)
plt.plot(timestamps[start:stop], csp_envelope[csp_component][start:stop], 
         label="envelope " + str(csp_component+1) + "/" + str(csp_filter_count), c="steelblue")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()


##### Example plots for first csp component #####
window_nr = 0


# outliers and pauses
outlier_data = [eeg_spatial_filtered[csp_component][i] for i in range(len(first_outliers)) if first_outliers[i]]
outlier_ts = [timestamps[i] for i in range(len(first_outliers)) if first_outliers[i]]
plt.plot(timestamps, eeg_spatial_filtered[csp_component], label="CSP component " + str(csp_component + 1), c="green")
plt.scatter(outlier_ts, outlier_data, label="outliers", c="red")
show_label = True
i = 0
while i < len(timestamps):
    if ts_info[i] == pause_id:
        x_left = timestamps[i]
        x_right = x_left
        is_cue = True
        while is_cue:
            i += 1
            if ts_info[i] == pause_id:
                is_cue = True
            else:
                is_cue = False
                x_right = timestamps[i]
                if show_label:
                    plt.axvspan(xmin = x_left, xmax = x_right, color="red", alpha=0.5, label="pauses")
                    show_label = False
                else:
                    plt.axvspan(xmin = x_left, xmax = x_right, color="red", alpha=0.5)
    i += 1
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# envelope
plt.plot(timestamps, first_envelope, label="envelope", c="steelblue")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# cumulated sum
plt.plot(timestamps, first_integrated, label="cumulated sum", c="orange")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# windows for cumulated sum
plt.axvline(first_win_ts[0][0], c="grey", label="windows")
for i in range(len(first_win_ts)):
    plt.axvline(first_win_ts[i][0], c="grey")
    plt.axvline(first_win_ts[i][-1], c="grey")
plt.plot(timestamps, first_integrated, label="cumulated sum", c="orange")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# windows for cumulated sum with outliers
a = len(first_win_ts)
b = len(first_win_ts[0])
outlier_windows = numpy.array(first_outliers[:a*b]).reshape(a, b)
outlier_windows = [numpy.any(outlier_windows[j]) for j in range(len(outlier_windows))]
plt.axvline(first_win_ts[0][0], c="grey", label="windows")
plt.axvspan(xmin = 0, xmax = 0, color="red", label="outlier or pause window")
width = first_win_ts[0][-1] - first_win_ts[0][0]
height = first_integrated[-1]
for i in range(len(first_win_ts)):
    x_left = first_win_ts[i][0]
    x_right = first_win_ts[i][-1]
    plt.axvline(x_left, c="grey")
    plt.axvline(x_right, c="grey")
    if outlier_windows[i]:
        plt.axvspan(xmin = x_left, xmax = x_right, color="red")
plt.plot(timestamps, first_integrated, label="cumulated sum", c="orange")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# windows for cumulative sum after outlier removal
plt.axvline(first_outlierfree_ts[0][0], c="grey", label="outlierfree windows")
plt.plot(first_outlierfree_ts[0], first_outlierfree[0], label="outlierfree cumulated sum", c="orange")
for i in range(len(first_outlierfree_ts)):
    plt.axvline(first_outlierfree_ts[i][0], c="grey")
    plt.axvline(first_outlierfree_ts[i][-1], c="grey")
    plt.plot(first_outlierfree_ts[i], first_outlierfree[i], c="orange")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# first window of cumulative sum
plt.plot(first_outlierfree_ts[window_nr], first_outlierfree[window_nr], 
         label="cumulated sum for window " + str(window_nr + 1), c="orange")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# first window detrended
plt.plot(first_win_ts[window_nr], first_outlierfree[window_nr], 
         label="cumulated sum for window " + str(window_nr + 1), c="orange")
plt.plot(first_win_ts[window_nr], first_fits[window_nr], label="linear fit", c="red")
plt.plot(first_win_ts[window_nr], first_detrended[window_nr], label="detrended", c="dodgerblue")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

# all windows detrended
plt.axvline(first_win_ts[0][0], c="grey", label="windows")
plt.axvline(first_win_ts[0][-1], c="grey")
plt.plot(first_win_ts[0], first_outlierfree[0], label="cumulated sum", c="orange")
plt.plot(first_win_ts[0], first_fits[0], label="linear fit", c="red")
plt.plot(first_win_ts[0], first_detrended[0], label="detrended", c="dodgerblue")
for i in range(1, len(first_outlierfree_ts)):
    plt.axvline(first_win_ts[i][0], c="grey")
    plt.axvline(first_win_ts[i][-1], c="grey")
    plt.plot(first_win_ts[i], first_outlierfree[i], c="orange")
    plt.plot(first_win_ts[i], first_fits[i], c="red")
    plt.plot(first_win_ts[i], first_detrended[i], c="dodgerblue")
plt.xlabel("time [sec]")
plt.ylabel("power [uV]")
plt.legend()
plt.show()

