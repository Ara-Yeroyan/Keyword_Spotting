import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def load_my_model():
	import tensorflow as tf
	from tensorflow.keras.models import model_from_json

	# json_file = open('./custom_enhanced.json', 'r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	from tensorflow.keras import layers, models
	model = models.Sequential([
	    layers.Input(shape=[124, 129, 1]),
	    # """
	    # layers.Conv2D(32, 5, strides = 2, activation='relu'),
	    # layers.Conv2D(32, 3, strides = 2, activation='relu'),
	    # layers.BatchNormalization(),
	    # layers.MaxPool2D(),
	    # preprocessing.Resizing(32, 32),
	    # """ 
	    #preprocessing.Resizing(32, 32)
	    layers.Conv2D(32, 5, strides = 2),
	    layers.Lambda(lambda x: tf.abs(x)),
	    layers.BatchNormalization(),
	    layers.ReLU(),
	    layers.MaxPool2D(),
	    layers.Conv2D(64, (3,3), activation='selu'),
	    layers.Conv2D(64, (3,3), activation = 'selu'),
	    layers.Conv2D(64, (3,3)),
	    layers.BatchNormalization(),
	    layers.ReLU(),
	    layers.MaxPooling2D(),
	    layers.Dropout(0.35),
	    layers.Flatten(),
	    layers.Dense(128, activation='selu'),
	    layers.Dropout(0.4),
	    layers.Dense(4, 'softmax'),
	])

	# load weights into new model
	model.load_weights("./custom_enhanced.h5")
	print("Loaded model from checkpoints !")

	return model
