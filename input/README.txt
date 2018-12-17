Functions for reading zip files into pandas dataframes
Data is contained in 10 zip files, in the data directory.
There is one zip file for each day containing a text file for that day.
There are 149 rows in each file: 40 LOB data points. 104 hand-crafted features and 5 prediction horizons.
Each column represents a snapshot of the 149 data points after every 10 messages 
Data is normalized based on prior day mean and standard deviation.
Data for each of 5 stocks is stored consecutively (i.e., first all of the snapshots for the first stock, etc.,)

