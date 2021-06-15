def record():
	import sounddevice as sd
	from scipy.io.wavfile import write
	import wavio as wv
	import librosa

	freq = 16000
	duration = 1
	recording = sd.rec(int(duration * freq),
					samplerate=freq, channels=1)

	sd.wait()
	wv.write("recording.wav", recording, freq, sampwidth=1)
	signal, rate = librosa.load("recording.wav", freq)
	return signal


# Convert the NumPy array to audio file
#wv.write("recording1.wav", recording, freq, sampwidth=2)
