import numpy as np
import pandas as pd
from io import StringIO
from zipfile import ZipFile

class DataLoader():
    def __init__(self,path):
        self.path=path
        self.zipfiles=['Day%i.zip' %i for i in range(1,11)]
        self.files=['Test_Dst_NoAuction_ZScore_CF_%i.txt' % i for i in range(1,10)]
        self.files.insert(0,'Train_Dst_NoAuction_ZScore_CF_1.txt')
        # offsets for each stock from beginning of that days data; last column is end of days data
        self.offsets=np.array([ 
                        [0, 3454,  9772, 14694, 25413, 39512],
                        [0, 5079, 11201, 17166, 23878, 38397],
                        [0, 3903,  9314, 13341, 18527, 28535],
                        [0, 2806,  9798, 15140, 25959, 37023],
                        [0, 3030,  9758, 15704, 22082, 34785],
                        [0, 2263,  8506, 14113, 21120, 39152],
                        [0, 2801,  9861, 16601, 24455, 37346],
                        [0, 2647, 11309, 19900, 33129, 55478],
                        [0, 1873, 11144, 21180, 34060, 52172],
                        [0, 1888,  7016, 12738, 18559, 31937]
                            ]) 
        self.diff=np.diff(self.offsets) # number of observations for each stock on each day
        

    
    # return dataframe for one day (rows are values,columns are times)
    #        
    def get_one_day(self,day):
        # days are indexed from 0 to 9
        z=self.path+self.zipfiles[day]
        zf=ZipFile(z,'r')
        data = StringIO(zf.read(self.files[day]).decode('utf_8'))
        df=pd.read_csv(data,sep='\s{2,}',engine='python',header=None)
        s=np.repeat([0,1,2,3,4],self.diff[day])
        return df
    
    # return index of stocks for list of days
    #
    def get_stock_index(self,days):
        s=[np.repeat([0,1,2,3,4],self.diff[day]) for day in days]
        return s
    
    # find offset (number of entries) from beginning of file to each stock on each selected day
    # this is needed to generate batches that keep stocks/days separate   
    def get_stock_offset(self,days):
        m=self.offsets[days,1:] #select days - drop off zero at beginning of each day
        n=self.offsets[days,-1].cumsum() # cumsum over number of entries in each selected day
        n=np.insert(n,0,0)[:-1].reshape(-1,1) # get index of cumsum up to beginning of each day
        return (m+n).reshape(-1) # return vector 
    
    #return  1) dataframe for multiple days - transpose so that columns are values and rows are time
    #        
    def get_days(self,days):
        df=[]
        for day in days:
            d=self.get_one_day(day)
            d=pd.DataFrame(d.values.T)
            df.append(d)
        df=pd.concat(df)
        return df
                
    
       