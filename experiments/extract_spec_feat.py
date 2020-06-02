import numpy as np
import pandas as pd


def get_wave_width_features(length, amp, argmax, max_amp_length, threshold):
    """
    ピーク値を有する山の両端の幅を得る
    """
    left_end = False
    right_end = False

    amp_under_threshold = (amp <= threshold)
    if argmax != 0:
        left = (length[:argmax])[amp_under_threshold[:argmax]]
        if not len(left) == 0:
            left_width = max_amp_length - left[-1]
        else:
            left_end = True
    if argmax != len(amp) - 1:
        right = (length[argmax:])[amp_under_threshold[argmax:]]
        if not len(right) == 0:
            right_width = right[0] - max_amp_length
        else:
            right_end = True

    # 見切れている場合は逆側と同じ長さにする
    if argmax == 0 or left_end:
        left_width = right_width
    if argmax == len(amp) - 1 or right_end:
        right_width = left_width

    return np.minimum(left_width, right_width), np.maximum(left_width, right_width)


def features_from_spectrogram(df_spec):
    length = df_spec["length"].values
    amp = df_spec["amp"].values
    argmax = np.argmax(amp)
    max_amp = amp[argmax]
    max_amp_length = length[argmax]
    min_amp = np.min(amp)
    mean_amp = np.mean(amp)
    std_amp = np.std(amp)
    p90_amp = np.percentile(amp, 90)

    width_75_min, width_75_max = get_wave_width_features(length, amp, argmax, max_amp_length, max_amp * 0.75)
    width_50_min, width_50_max = get_wave_width_features(length, amp, argmax, max_amp_length, max_amp * 0.50)

    return max_amp, max_amp_length, width_75_min, width_75_max, width_50_min, width_50_max, min_amp, mean_amp, std_amp, p90_amp


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
file_path = '../input/spectrum_raw/'

additional_features_train = []
additional_features_test = []

for i, file_name in enumerate(train["spectrum_filename"]):
    df_spec = pd.read_csv(file_path + file_name, sep="\t", header=None, names=('length', 'amp'))
    features = features_from_spectrogram(df_spec)
    additional_features_train.append(features)

for i, file_name in enumerate(test["spectrum_filename"]):
    df_spec = pd.read_csv(file_path + file_name, sep="\t", header=None, names=('length', 'amp'))
    features = features_from_spectrogram(df_spec)
    additional_features_test.append(features)

additional_features_train = pd.DataFrame(additional_features_train)
additional_features_test = pd.DataFrame(additional_features_test)
additional_features_train.columns = [
    'max_amp', 'max_amp_length', 'width_75_min', 'width_75_max', 'width_50_min',
    'width_50_max', 'min_amp', 'mean_amp', 'std_amp', 'p90_amp'
]
additional_features_test.columns = [
    'max_amp', 'max_amp_length', 'width_75_min', 'width_75_max', 'width_50_min',
    'width_50_max', 'min_amp', 'mean_amp', 'std_amp', 'p90_amp'
]

additional_features_train.to_csv('../input/additional_features_train.csv', index=False)
additional_features_test.to_csv('../input/additional_features_test.csv', index=False)
