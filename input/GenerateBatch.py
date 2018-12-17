import math
import numpy as np
import pandas as pd

class GenerateBatch():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self,datx,daty,idx):
        self.datx = datx
        self.daty = daty
        self.len= self.datx.shape[0]
        self.idx=idx

    # epoch_size rounds up the number of batches for each stock/day combination and sums them
    def epoch_size(self, seq_len,batch_size):
         b=np.diff(np.insert(self.idx,0,0))
         return np.sum([np.ceil((i-seq_len)/batch_size) for i in b])
       
    def GenerateBatch(self, seq_len, batch_size):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i <= (self.len - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if (i <=(self.len - seq_len)): # short batch if reach end of data
                    x, y = self._next_window(i, seq_len)
                    x_batch.append(x)
                    y_batch.append(y)
                    i+=1
 
            if i>(self.len-seq_len):i=0 # batch used up all data - start again
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len):
        '''Generates the next data window from the given index location i'''
        x = self.datx[i:i+seq_len]
        y = self.daty[i+seq_len-1]
        return x, y

    def GenerateBatch2(self, seq_len, batch_size,nflag=False):
        '''Yield a generator of training data from filename on given list of cols split for train/test
           Does not allow a single batch to include more than one stock/day combination
           nflag - normalize each window by subtracting first row 
        
        '''
        i = 0 #index into data
        j=0 # index into list of breaks for stocks
        while i <= (self.len - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                # stop appending to the batch if reached end of file or reached end of data for one day's stock data
                if (i <=(self.len - seq_len))&(i<=(self.idx[j]-seq_len)): 
                        x, y = self._next_window(i, seq_len)
                        if nflag==True:
                            x=x-x[0,:]
                        x_batch.append(x)
                        y_batch.append(y)
                        i+=1
            if i>(self.len-seq_len): 
                   i=0 # start at beginning
                   j=0
            if i>(self.idx[j]-seq_len): 
                   i=i+seq_len-1
                   j=j+1 #continue next batch with next stock
            yield np.array(x_batch), np.array(y_batch) # yield current batch
            
    def GenerateBatch_conv(self, seq_len, batch_size,nflag=False):
        '''Yield a generator of training data from filename on given list of cols split for train/test
           Does not allow a single batch to include more than one stock/day combination
           Add 1 dimension for channels - for convnets
           nflag - normalize each window by subtracting first row 
        '''
        i = 0 #index into data
        j=0 # index into list of breaks for stocks
        while i <= (self.len - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                # stop appending to the batch if reached end of file or reached end of data for one day's stock data
                if (i <=(self.len - seq_len))&(i<=(self.idx[j]-seq_len)): 
                        x, y = self._next_window(i, seq_len)
                        x_batch.append(x)
                        if nflag==True:
                            x=x-x[0,:]
                        y_batch.append(y)
                        i+=1
            if i>(self.len-seq_len): 
                   i=0 # start at beginning
                   j=0
            if i>(self.idx[j]-seq_len): 
                   i=i+seq_len-1
                   j=j+1 #continue next batch with next stock
            #yield current batch add 1 dim for channels   
            yield np.expand_dims(np.array(x_batch),3),np.array(y_batch)

