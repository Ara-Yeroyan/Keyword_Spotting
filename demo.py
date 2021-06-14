import numpy as np
import pandas as pd
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import io, warnings, sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import tensorflow as tf 
from preprocessing import *
from load_model import *
from record_audio import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:  print('working on CPU')

from inference import inspection
import matplotlib.pyplot as plt


import tensorflow.keras.backend as K
@st.cache(allow_output_mutation=True)
def loading_model():
	model = load_my_model()
	#model._make_predict_function()
	model.summary()  # included to make it visible when model is reloaded
	return model

def analysis(signal, r = 16000, label = 'recording'):
			raw = tf.convert_to_tensor(signal)
			#waveform, label = get_waveform_and_label(uploaded_file, True)
			spectrogram = get_spectrogram(raw)
			log_spec = np.log(spectrogram.numpy().T)

			if st.checkbox('Raw Data'):
				time = r / 16000
				st.write(f'Audio lenght: {time} seconds')
				pd.Series(signal).plot(figsize=(8,3))
				plt.xlabel('Time')
				plt.ylabel('Amplitude')
				plt.title('Time Domain')
				plt.xlim([0, 16000])
				st.pyplot()

			if st.checkbox('Transformed Data'): 

				plot_fft(signal,r)
				st.pyplot()
			if st.checkbox('Spectrogram'): 

				st.write(f"spectrogram's shape: {spectrogram.shape}")
				#fig, axes = plt.subplots(2, figsize=(12, 8))
				# timescale = np.arange(signal.shape[0])
				# axes[0].plot(timescale, signal)
				# axes[0].set_title('Waveform')
				# axes[0].set_xlim([0, 16000])
				plot_spectrogram(spectrogram.numpy(), plt)
				plt.title(label)
				plt.show()	
				st.pyplot()

			st.subheader('Time for Prediction')
			if st.button('Load and Predict'):
				model = loading_model()
				st.write('Succesfully loaded the model !')
				decoding = ['no' , 'noise', 'unknown', 'yes']
				batch = spectrogram.numpy().reshape(1, spectrogram.shape[0], spectrogram.shape[1], 1)
				pred = model.predict_classes(batch)
				pred_text = decoding[pred[0]]
				#print(type(pred.value))
				st.write(f'The given audio belongs to **{pred_text}**: class')
				df = pd.read_csv('track.csv')
				df.iloc[0,0] = 0
				df.to_csv('track.csv', index = False)

@st.cache()
def record_once():
	return record()

"""Simple Keyword Trigerr"""

st.title("Keyword Spotting")
st.sidebar.markdown('# Choose The Section')
navigation = st.sidebar.radio('How to input the audio', ('Upload a ".wav" file', 'Manually Record'))


import librosa
df = pd.read_csv('track.csv')
signal = np.array(df.iloc[0, 1:])    #_ = librosa.load('yes_recording_final.wav', 16000)
if navigation != 'Upload a ".wav" file':
	st.subheader('Record your audio')
	if st.checkbox('Start'):

		df = pd.read_csv('track.csv')
		tracker = df.iloc[0, 0]
		if tracker == 0:
			df.iloc[0, 0] =  1
			signal = record()
			df.iloc[0, 1:] = signal
			st.subheader('Inspect your recording !')
			st.write('Waveform shape:', signal.shape)
			st.audio('recording.wav')
			df.to_csv('track.csv', index = False)
			#from IPython import display	
			#display.display(display.Audio(signal, rate=16000))

	st.subheader('Analyse your data')
	choice = st.selectbox('Start ?', ('No', 'Yes'), key = 'Analise')
	if choice == 'Yes':
		analysis(signal)

else:
	uploaded_file = st.sidebar.file_uploader("Upload your CSV file with data here", type=["wav"])
	if uploaded_file is not None:
		audio = uploaded_file.name
		import regex as re
		pattern = r'\w+'
		res = re.findall(pattern, audio)
		label = res[0]
		st.success('Successfully read the audio: ' + audio)

		import librosa
		signal,r = librosa.load(uploaded_file, 16000)

		analysis(signal, label = label)

		


