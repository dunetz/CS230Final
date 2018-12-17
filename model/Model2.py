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
        self.tbCallback=callbacks.TensorBoard(log_dir='./Graph',histogram_freq=0,write_graph=True, write_images=True)
        

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
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()       
#
# define model directly without config file
#
    def build_model2(self):
        print("build_model2")
        self.model.add(LSTM(100,input_shape=(20,40),kernel_regularizer=regularizers.l2(0.01), return_sequences=False))
        self.model.add(Dropout(0.20))
        #self.model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
        #self.model.add(Dropout(0.20))
        self.model.add(Dense(3,activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam")
#
#changes dropout  versus build_model2 
#
    def build_model3(self):
        print("build_model2")
        self.model.add(LSTM(100,input_shape=(20,40),kernel_regularizer=regularizers.l2(0.01), return_sequences=False))
        self.model.add(Dropout(0.50))
        #self.model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
        #self.model.add(Dropout(0.20))
        self.model.add(Dense(3,activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

#
# first conv2d model
#
    def build_conv2d_model1(self,window,learning_rate,learning_rate_decay):
        print("build_conv2d_model1")
        self.model.add(Conv2D(16, kernel_size=(4, 40), strides=(1, 1),data_format='channels_last',activation='relu',input_shape=(window,40,1)))
        self.model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(Conv2D(16,kernel_size=(4,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,1), strides=2))
        self.model.add(Conv2D(32,kernel_size=(3,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,1), strides=2))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.50))
        #self.model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
        #self.model.add(Dropout(0.20))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])

#
# conv2d model + batch normalization
#
    def build_conv2d_bn_model1(self,window,learning_rate,learning_rate_decay):
        m=.9
        print("build_conv2d_model1")
        self.model.add(Conv2D(16, kernel_size=(4, 40), strides=(1, 1),data_format='channels_last',activation='relu',input_shape=(window,40,1)))
        self.model.add(BatchNormalization(momentum=m))
        self.model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(BatchNormalization(momentum=m))
        self.model.add(Conv2D(16,kernel_size=(4,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(BatchNormalization(momentum=m))
        self.model.add(MaxPooling2D(pool_size=(2,1), strides=2))
        self.model.add(Conv2D(32,kernel_size=(3,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(BatchNormalization(momentum=m))
        self.model.add(MaxPooling2D(pool_size=(2,1), strides=2))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.50))
        #self.model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01),activation='relu'))
        #self.model.add(Dropout(0.20))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])



#
# first conv2d-LSTM model
#
    def build_conv2d_LSTM_model1(self,window,learning_rate,learning_rate_decay):
        
        def squeeze_axis(x):
            xS= squeeze(x,axis=2) # squeeze channel dimension
            #xT = permute_dimensions(xT,(0,2,1)) # axis 1 and 2 are swapped    
            return xS
    
        print("build_conv2d_model1")
        self.model.add(Conv2D(16, kernel_size=(4, 40), strides=(1, 1),data_format='channels_last',activation='relu',input_shape=(window,40,1)))
        self.model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(Conv2D(16,kernel_size=(4,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,1), strides=2))
        self.model.add(Conv2D(32,kernel_size=(3,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(Lambda(squeeze_axis))
        self.model.add(LSTM(64, activation='relu',return_sequences=False))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])
        
#
# second conv2d-LSTM model - add dropout - replace dropout with l2 regularizer
#
    def build_conv2d_LSTM_model2(self,window,learning_rate,learning_rate_decay):
        
        def squeeze_axis(x):
            xS= squeeze(x,axis=2) # squeeze channel dimension
            #xT = permute_dimensions(xT,(0,2,1)) # axis 1 and 2 are swapped    
            return xS
    
        print("build_conv2d_model1")
        self.model.add(Conv2D(16, kernel_size=(4, 40), strides=(1, 1),data_format='channels_last',activation='relu',input_shape=(window,40,1)))
        self.model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(Conv2D(16,kernel_size=(4,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,1), strides=2))
        self.model.add(Conv2D(32,kernel_size=(3,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(Lambda(squeeze_axis))
        self.model.add(LSTM(64, activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=False))
        #self.model.add(Dropout(0.40))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])
        
#
# second conv2d-LSTM model - Second LSTM layer
#
    def build_conv2d_LSTM_model3(self,window,learning_rate,learning_rate_decay):
        
        def squeeze_axis(x):
            xS= squeeze(x,axis=2) # squeeze channel dimension
            #xT = permute_dimensions(xT,(0,2,1)) # axis 1 and 2 are swapped    
            return xS
    
        print("build_conv2d_model1")
        self.model.add(Conv2D(16, kernel_size=(4, 40), strides=(1, 1),data_format='channels_last',activation='relu',input_shape=(window,40,1)))
        self.model.add(Conv2D(16, kernel_size=(1,1),strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(Conv2D(16,kernel_size=(4,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,1), strides=2))
        self.model.add(Conv2D(32,kernel_size=(3,1), strides=(1,1),data_format='channels_last',activation='relu'))
        self.model.add(Lambda(squeeze_axis))
        self.model.add(LSTM(100, activation='relu',return_sequences=True))
        self.model.add(LSTM(100, activation='relu',return_sequences=False))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])
        
#
# simple Conv1D Model
# remove first maxpool
# add 1x1 
    def build_conv1d_model1(self,window,learning_rate,learning_rate_decay):
        print("build_conv1d_model1")
        self.model.add(Conv1D(16, kernel_size=4, strides=1,activation='relu',input_shape=(window,40)))    
        self.model.add(Conv1D(16, kernel_size=1, strides=1,activation='relu'))   
        #self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(16,kernel_size=4,activation='relu'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(32,kernel_size=3,activation='relu'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Flatten())        
        #self.model.add(Dense(100,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dropout(0.50))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])
        
#
# Conv1D LSTM-Model
#
    def build_conv1d_LSTM_model1(self,window,learning_rate,learning_rate_decay):
        print("build_conv1d_model1")
        self.model.add(Conv1D(16, kernel_size=4, strides=1,activation='relu',input_shape=(window,40)))    
            
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(16,kernel_size=4,activation='relu'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(32,kernel_size=3,activation='relu'))
        self.model.add(MaxPooling1D(2))
        self.model.add(LSTM(64,activation='relu',kernel_regularizer=regularizers.l2(0.01),return_sequences=False))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])


        
#
# conv1d model with dilation
#
    def build_conv1d_dilation_model1(self):
        print("build_conv1d_model1")
        self.model.add(Conv1D(16, kernel_size=2, strides=1,activation='relu',padding='causal',input_shape=(100,40)))    
            
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=2,activation='relu',padding='causal'))
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=4,activation='relu',padding='causal'))
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=8,activation='relu',padding='causal'))
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=16,activation='relu',padding='causal'))
        self.model.add(Flatten())        
        self.model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(Dropout(0.40))
        #self.model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        #self.model.add(Dropout(0.40))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])

#
# second conv1d model - with LSTM at end
# add window size, learning rate and learing rate decay as parameters, 
# not yet implemented - add 1x1 convolution to reduce number of parameters
    def build_conv1d_model2(self,window,learning_rate,learning_rate_decay):
        print("build_conv1d_model1")
        self.model.add(Conv1D(16, kernel_size=2, strides=1,activation='relu',padding='causal',input_shape=(window,40)))    
            
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=2,activation='relu',padding='causal'))
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=4,activation='relu',padding='causal'))
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=8,activation='relu',padding='causal'))
        self.model.add(Conv1D(16,kernel_size=2,dilation_rate=16,activation='relu',padding='causal'))
        #1x1 convolution to restore to one channel
        #self.model.add(Conv1D(1,kernel_size=1)
        self.model.add(LSTM(64, activation='relu',return_sequences=False))
        #self.model.add(Flatten())
        #self.model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        #self.model.add(Dropout(0.40))
        #self.model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
        #self.model.add(Dropout(0.40))
        self.model.add(Dense(3, activation='softmax'))
        opt=Adam(lr=learning_rate,decay=learning_rate_decay)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])
      
       

    def train(self, x, y, epochs, batch_size, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [self.tbCallback,
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def train_generator(self, train_gen, val_gen, epochs, batch_size, steps_per_epoch, validation_steps, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [self.tbCallback,
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        history=self.model.fit_generator(
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
    
    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
