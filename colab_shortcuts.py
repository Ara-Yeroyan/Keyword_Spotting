import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, warnings, sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import tensorflow as tf

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]  # working with tensorflow graph

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def plot_signals(audios, random = 0):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False,
                             sharey=True, figsize=(25, 1))
    #fig.suptitle('Time Series', size= 20)
    for y in range(4):
      wav, label = get_waveform_and_label(audios[(1500*y-1)-30*random])
      label = label.numpy().decode('utf-8')
      axes[y].set_title(label, family ='serif');
      axes[y].plot(wav)
      axes[y].get_xaxis().set_visible(False)
      axes[y].get_yaxis().set_visible(False)

from librosa import display as Display
def plot_log_spectograms(path_to_audios):
  fig, axs = plt.subplots(1, 4, figsize = (25, 4))
  for i in range(4):
    signal , rate = librosa.load(path_to_audios[np.random.randint(6036)], sr = 16000) 
    stft = librosa.core.stft(signal,hop_length= 128, n_fft = 256)
    spectogram = np.abs(stft) # we are working with complex numbers, need to get magnitudes
    log_spectogram = librosa.amplitude_to_db(spectogram)
    #assert log_spectogram.shape == (513, 32)
    z = Display.specshow(log_spectogram, hop_length= 128, sr = rate, ax = axs[i])
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Magnitude")
    axs[i].set_title(path_to_audios[np.random.randint(6036)].split('/')[-1])
    plt.colorbar(z, ax = axs[i])

# import tensorflow.experimental.numpy as tnp
# tnp.experimental_enable_numpy_behavior()
def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=256, frame_step=128)

  #a,b, spectrogram = scipy.signal.stft(equal_length.numpy(), noverlap = 128, nperseg=256)
  
  spectrogram = tf.transpose(tf.abs(spectrogram))
  #spectrogram = tf.math.log(spectrogram)

  return spectrogram

def plot_spectrogram(spectrogram, ax):
  log_spec = np.log(spectrogram)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


def get_spectrogram_and_label_id(audio, label):
  labels = ['no' , 'noise', 'unknown', 'yes']
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == labels)
  return spectrogram, label_id

def visualise_spectogram_data(rows, cols, spectrogram_ds):
  labels = ['no' , 'noise', 'unknown', 'yes']
  n = rows*cols
  fig, axes = plt.subplots(rows, cols, figsize=(25, 6))
  for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
    ax.set_title(labels[label_id.numpy()])
    ax.axis('off')
    
  plt.show()

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds