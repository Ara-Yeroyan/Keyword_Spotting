import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path, colab = True):
  if colab:
    parts = tf.strings.split(file_path, os.path.sep)[-2]
  else:
    parts = tf.strings.split(file_path.split('/')[-2]).numpy()[0]
  return parts

def get_waveform_and_label(file_path, no_label = False):
  if no_label: label = None
  else: label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Padding for files with less than 1 second
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=256, frame_step=128)
  
  spectrogram = tf.abs(spectrogram)

  return tf.transpose(spectrogram)

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  #spectrogram = spectrogram.reshape((124, 129))
  log_spec = np.log(spectrogram)
  print(f'log spec shape: {log_spec.shape}')
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = np.arange(height)
  ax.pcolormesh(X, Y, log_spec)
  #ax.colorbar()


def get_spectrogram_and_label_id(audio, no_label = False):
  labels = ['no' , 'noise', 'unknown', 'yes']
  waveform, label = get_waveform_and_label(audio)
  #label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)
  spectrogram = tf.expand_dims(spectrogram, -1)
  if no_label: label_id = None
  else: label_id = np.argmax(label.decode('utf-8') == labels)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds








