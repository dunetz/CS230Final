'''
 This version hardcodes l2 regularization for all layers except softmax
 Should be parameterized for config file
'''


import os
import math
import numpy as np
import datetime as dt
from keras import regularizers
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt

class Timer():

   def __init__(self):
      self.start_dt = None

   def start(self):
      self.start_dt = dt.datetime.now()

   def stop(self):
      end_dt = dt.datetime.now()
      print('Time taken: %s' % (end_dt - self.start_dt))
      
class Model():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):
        timer = Timer()
        timer.start()
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                if activation=='softmax':
                    self.model.add(Dense(neurons, activation=activation))
                else:               
                    self.model.add(Dense(neurons, activation=activation, kernel_regularizer=regularizers.l2(0.01)))
            if layer['type'] == 'lstm':
                    self.model.add(LSTM(neurons, kernel_regularizer=regularizers.l2(0.01), input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def build_model2(self):
        timer = Timer()
        timer.start()
        self.model.add(LSTM(50, kernel_regularizer=regularizers.l2(0.01), input_shape=(10,40), return_sequences=False))
        self.model.add(Dropout(0.20))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        print('[Model] Model Compiled')
        timer.stop()
        
    def build_model3(self):
        timer = Timer()
        timer.start()
        self.model.add(LSTM(50, kernel_regularizer=regularizers.l2(0.01), input_shape=(10,40), return_sequences=True))
        self.model.add(Dropout(0.20))
        self.model.add(LSTM(50, kernel_regularizer=regularizers.l2(0.01), input_shape=(50), return_sequences=False))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
        print('[Model] Model Compiled')
        timer.stop()


    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            size=batch_size,
            cks=callbacks
		)
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, train_gen, val_gen, epochs, batch_size, steps_per_epoch,validation_steps, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
        history = self.model.fit_generator(
			train_gen,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			verbose=2,
              validation_data=val_gen,
              validation_steps=validation_steps,
			callbacks=callbacks,
			workers=1
		)
		
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
        return history
	
