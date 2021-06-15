import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

def read_audio(audio_path):
  audio, rate = librosa.load(audio_path, 16000)
  assert rate == 16000
  return audio

import regex as re
def extract_label(file_path):
    pattern = r'\w+'
    res = re.findall(pattern, file_path)
    return res[0]  # working with tensorflow graph

def get_waveform_and_label_numpy(file_path):
  label = extract_label(file_path)
  waveform = read_audio(file_path)
  return waveform, label


def calc_fft(signal, rate = 16000):

  n = len(signal)
  # rfft for avoining "Hamaluster"  (Complex numbers are included)
  freq = np.fft.rfftfreq(n, d=1/rate)  # d -> the spacing between each individual samples
  # abs -> for getting the magnitude (in complex numbers sqrt(Re**2 + Im**2))  -> spectrum
  y = abs(np.fft.rfft(signal)/n) 
  return (y, freq)


def plot_fft(signal, rate = 16000):

  Y, freq = calc_fft(signal, rate)
  plt.plot(freq, Y)
  plt.title('Fourier Domain')
  #plt.get_yaxis().set_visible(False)
  plt.show()


def plot_signals(audios, random = 0):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False,
                             sharey=True, figsize=(25, 1))
    #fig.suptitle('Time Series', size= 20)
    for y in range(4):
      wav, label = get_waveform_and_label_numpy(audios[(1500*y-1)-30*random])
      label = label.numpy().decode('utf-8')
      axes[y].set_title(label, family ='serif');
      axes[y].plot(wav)
      axes[y].get_xaxis().set_visible(False)
      axes[y].get_yaxis().set_visible(False)
     
def get_spectrogram_scipy(waveform : np.array):
  import scipy
  # Padding for files with less than 16000 samples
  zero_padding = np.zeros(16000 - waveform.shape[0], dtype=np.float32)

  waveform = waveform.astype(np.float32)
  equal_length = np.concatenate([waveform, zero_padding], 0)
  # spectrogram = tf.signal.stft(
  #     equal_length, frame_length=256, frame_step=128)
  _, _ , spectrogram = scipy.signal.stft(equal_length, noverlap = 128, nperseg=256)
  
  spectrogram = np.abs(spectrogram)

  return spectrogram

def plot_spectrogram_scipy(spectrogram, ax):
  log_spec = np.log(spectrogram)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

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

def visualise_spectogram_data(rows, cols, audios):
  labels = ['no' , 'noise', 'unknown', 'yes']
  n = rows*cols
  fig, axes = plt.subplots(rows, cols, figsize=(25, 6))
  for i in range(n):
    idx = np.random.randint(len(audios))
    target = audios[i]
    waveform, label = get_waveform_and_label_numpy(target)
    spectrogram = get_spectrogram_scipy(waveform)
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(spectrogram, ax)
    ax.set_title(label)
    ax.axis('off')
    
  plt.show()
