import librosa
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import roc_curve, auc

def extract_signal_features(
    signal,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=80,
    frames=5
):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    dims = frames * n_mels

    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)

    features = np.zeros((features_vector_size, dims), np.float32)
    for time in range(frames):
        features[:, n_mels * time : n_mels * (time + 1)] = log_mel_spectrogram[
            :, time : time + features_vector_size
        ].T

    return features


def generate_dataset(files_list, n_fft=1024, hop_length=512, n_mels=128, frames=5):
    dims = n_mels * frames

    for index in tqdm(range(len(files_list))):
        signal, sr = load_sound_file(files_list[index])
        features = extract_signal_features(
            signal,
            sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            frames=frames,
        )

        if index == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)

        dataset[
            features.shape[0] * index : features.shape[0] * (index + 1), :
        ] = features

    return dataset


def load_sound_file(path):
    signal, sr = librosa.load(path, sr=None)
    sound_file = signal, sr

    return sound_file


def compute_partial_auc(y_true, y_scores, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Keep only points where FPR <= max_fpr
    mask = fpr <= max_fpr
    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]

    # Interpolate to add (max_fpr, interpolated_tpr) if needed
    if fpr_partial[-1] < max_fpr:
        # Find next point beyond max_fpr
        idx = np.searchsorted(fpr, max_fpr)
        fpr_left, fpr_right = fpr[idx - 1], fpr[idx]
        tpr_left, tpr_right = tpr[idx - 1], tpr[idx]

        # Linear interpolation
        slope = (tpr_right - tpr_left) / (fpr_right - fpr_left)
        tpr_interp = tpr_left + slope * (max_fpr - fpr_left)

        fpr_partial = np.append(fpr_partial, max_fpr)
        tpr_partial = np.append(tpr_partial, tpr_interp)

    return auc(fpr_partial, tpr_partial)


def extract_log_mel_windows_VAE(
    file_path,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=80,
    frames=5
):
    signal, _ = librosa.load(file_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize each log-mel spectrogram to [0, 1]
    #log_mel_spec -= log_mel_spec.min()
    #log_mel_spec /= (log_mel_spec.max() + 1e-6)  # avoid division by zero

    total_frames = log_mel_spec.shape[1]
    num_windows = total_frames - frames + 1
    if num_windows < 1:
        return np.empty((0, frames, n_mels, 1), dtype=np.float32)

    windows = []
    for i in range(num_windows):
        window = log_mel_spec[:, i:i+frames]     # shape: (128, frames)
        window = window.T                        # shape: (frames, 128)
        window = np.expand_dims(window, axis=-1)  # shape: (frames, 128, 1)
        windows.append(window)

    return np.array(windows, dtype=np.float32)

def generate_dataset_from_list_VAE(
    file_list,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=80,
    frames=5
):
    all_features = []
    for file_path in tqdm(file_list, desc="Extracting features"):
        feats = extract_log_mel_windows_VAE(
            file_path, sr=sr,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, frames=frames
        )
        if feats.size > 0:
            all_features.append(feats)

    if all_features:
        return np.concatenate(all_features, axis=0)
    else:
        return np.empty((0, frames, n_mels, 1), dtype=np.float32)




def extract_log_mel_windows_LSTM(
    file_path,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    frames=5
):
    signal, _ = librosa.load(file_path, sr=sr)

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    total_frames = log_mel_spec.shape[1]
    num_windows = total_frames - frames + 1

    if num_windows < 1:
        return np.empty((0, frames, n_mels), dtype=np.float32)

    windows = []
    for i in range(num_windows):
        window = log_mel_spec[:, i:i+frames]   # shape: (128, frames)
        window = window.T                      # shape: (frames, 128)
        windows.append(window)

    return np.array(windows, dtype=np.float32)

def generate_dataset_from_list_LSTM(
    file_list,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    frames=5
):
    all_features = []
    for file_path in tqdm(file_list, desc="Extracting features"):
        feats = extract_log_mel_windows(
            file_path, sr=sr,
            n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, frames=frames
        )
        if feats.size > 0:
            all_features.append(feats)

    if all_features:
        return np.concatenate(all_features, axis=0)  # shape: (samples, frames, n_mels)
    else:
        return np.empty((0, frames, n_mels), dtype=np.float32)
