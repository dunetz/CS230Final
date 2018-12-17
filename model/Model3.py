import os
import math
import numpy as np
import datetime as dt
from keras import regularizers, callbacks
from keras.layers import Dense, Activation, Dropout, LSTM,Lambda 
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Reshape, BatchNormalization
from keras.backend import transpose, permute_dimensions,squeeze
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime as dt

class Timer():

   def __init__(self):
      self.start_dt = None

   def start(self):
      self.start_dt = dt.datetime.now()

   def stop(self):
      end_dt = dt.datetime.now()
      print('Time taken: %s' % (end_dt - self.start_dt))
      

def train_generator(model, train_gen, val_gen, epochs, batch_size, steps_per_epoch, validation_steps, save_dir):
   timer = Timer()
   timer.start()
   print('[Model] Training Started')
   print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
   tbCallback=TensorBoard(log_dir='./Graph',histogram_freq=0,write_graph=True, write_images=True)
   save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
   callbacks = [tbCallback,
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
   history=model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            workers=1
        )
        
   print('[Model] Training Completed. Model saved as %s' % save_fname)
   timer.stop()
   return history
    
    
