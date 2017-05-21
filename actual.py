#Used for numpy arrays
import numpy as np
#Used to read data from CSV file
import pandas as pd
#Used to convert date string to numerical value
from datetime import datetime, timedelta
#Used to plot data
import matplotlib.pyplot as mpl
#Used to scale data
from sklearn.preprocessing import StandardScaler
#Used to perform CV
from sklearn.cross_validation import KFold
from TFMLP import MLPR
from sklearn.neighbors import KNeighborsRegressor
#Used to get command line arguments
import sys

#Gives a list of timestamps from the start date to the end date
#
#startDate:     The start date as a string xxxx-xx-xx 
#endDate:       The end date as a string year-month-day
#period:        'daily', 'weekly', or 'monthly'
#weekends:      True if weekends should be included; false otherwise
#return:        A numpy array of timestamps      
def DateRange(startDate, endDate, period, weekends = True):
    #The start and end date
    sd = datetime.strptime(startDate, '%Y-%m-%d')
    ed = datetime.strptime(endDate, '%Y-%m-%d')
    #Invalid start and end dates
    if(sd > ed):
        raise ValueError("The start date cannot be later than the end date.")
    #One time period is a day
    if(period == 'daily'):
        prd = timedelta(1)
    #One prediction per week
    elif(period == 'weekly'):
        prd = timedelta(7)
    #one prediction every 30 days ("month")
    else:
        prd = timedelta(30)
    #The final list of timestamp data
    dates = []
    cd = sd
    while(cd <= ed):
        #If weekdays are included or it's a weekday append the current ts
        if(weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            dates.append(cd.timestamp())
        #Onto the next period
        cd = cd + prd
    return np.array(dates)
    
#Given a date, returns the previous day
#
#startDate:     The start date as a datetime object
#weekends:      True if weekends should counted; false otherwise
def DatePrevDay(startDate, weekends = False):
    #One day
    day = timedelta(1)
    cd = datetime.fromtimestamp(startDate)
    while(True):
        cd = cd - day
        if(weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            return cd.timestamp()
    #Should never happen
    return None

#Load data from the CSV file. Note: Some systems are unable
#path:      The path to the file
#return:    A data frame with the parsed timestamps
#path = '/home/mai/Desktop/actualtest/openhab.csv'
def ParseData(p):
    #Read the csv file into a dataframe
    df = pd.read_csv(p)
    #Get the date strings from the date column
    dateStr = df['Date'].values
    D = np.zeros(dateStr.shape)
    #Convert all date strings to a numeric value
    for i, j in enumerate(dateStr):
        #Date strings are of the form year-month-day 
        D[i] = datetime.strptime(j, '%Y-%m-%d').timestamp()
    #Add the newly parsed column to the dataframe
    df['Timestamp'] = D
    #Remove any unused columns (axis = 1 specifies fields are columns)
    return df.drop('Date', axis = 1) 

   
    '''
    #Transform the data (only used once)
    newdate = ["" for x in range(21723)]
    datevalues = df['Date'].values
    tempint = df['Temp'].values
    print (df.dtypes)
    
    
    addition = np.zeros(dateStr.shape, dtype=np.float64)
    countarray = np.zeros(dateStr.shape, dtype=np.float64) 
    averagearray = np.zeros(dateStr.shape, dtype=np.float64)  
    
    datecol = df['Date'].shape[0]
    
    count = 1
    j = 0
    
    addition[0] = tempint[0]
    newdate [0]= datevalues[0]

    #print ('newdate of zero ',newdate[0])
 
    for i in range (datecol-1):

        prevday = df['Date'][i]
        #prevtemp = df['Temp'][i]
        prevtemp = tempint[i]
        char1 = prevday[8]
        char2 = prevday[9]
        c1 = float(char1)
        c2 = float (char2)
        sum = c1 + c2

        #currentdate = df['Date'].loc[i+1]
        currentdate = datevalues[i+1]
        #currenttemp = df['Temp'].loc[i+1]
        currenttemp = tempint[i+1]
        #parsedcurrenttemp = float (currenttemp)
        char3 = currentdate[8]
        char4 = currentdate[9]
        c3= float(char3)
        c4 = float(char4)
        sum2 = c3 + c4
        
        #print ('sum 1:',sum , 'sum 2:', sum2)
        if (sum == sum2):
            #parsedavg = float(average[j])
            #average [j] = parsedavg + parsedcurrenttemp
            #average [j] = average[j] + currenttemp

            newdate[j] = currentdate
            addition [j]= addition [j]+ currenttemp 
            count = count +1
            countarray [j]= count 
            #print ('j:', j )
            #average [j]= average[j] / count 
            #count = count +1 
            #print ('Iteration number:', i , 'Addition:', addition[i]) 
        else :
            count = 1
            j = j+1
            #average [j] = average[j] + currenttemp / count 
            addition [j]= addition[j]+currenttemp
            newdate[j] = currentdate


    print ('Date array:',newdate[10])
    print ('Additionarray',addition[0])
    print ('Countarray', countarray[0])
    
    for i in range(len(newdate)):
        averagearray[i] = addition[i] / countarray[i]

    print ('Avgarray', averagearray)

    Header1 = ["Date","Temp","Average"]
    df = pd.DataFrame({'Date':newdate,'Temp':addition,'Average':averagearray})
    df.to_csv('/home/mai/Desktop/Outputs/actual7.csv',columns = Header1)
    df['NewDate'] = newdate 
    df['NewTemp'] = addition
    #df.drop('Date', axis = 1) 
    #df.drop('Temp', axis = 2)
    #print (df)
'''
    
    
    
        
#Given dataframe from ParseData
#plot it to the screen
#
#df:        Dataframe returned from 
#p:         The position of the predicted data points
def PlotData(df, p = None):
    if(p is None):
        p = np.array([])
    #Timestamp data
    ts = df.Timestamp.values

    #Number of x tick marks
    nTicks= 10
    #Left most x value
    s = np.min(ts)
    #Right most x value
    e = np.max(ts)
    #Total range of x values
    r = e - s
    #Add some buffer on both sides
    s -= r / 5
    e += r / 5
    #These will be the tick locations on the x axis
    tickMarks = np.arange(s, e, (e - s) / nTicks)
    #Convert timestamps to strings
    strTs = [datetime.fromtimestamp(i).strftime('%m-%d-%y') for i in tickMarks]
    mpl.figure()
    #Plots of the high and low values for the day
    mpl.plot(ts, df.Temp.values, color = '#727272', linewidth = 1.618, label = 'Actual')
    #Predicted data was also provided
    if(len(p) > 0):
        mpl.plot(ts[p], df.Temp.values[p], color = 'r', linewidth = 1.618, label = 'Predicted')
    #Set the tick marks
    mpl.xticks(tickMarks, strTs, rotation='vertical')
    #Set y-axis label
    mpl.ylabel('Tempreature')
    #Add the label in the upper left
    mpl.legend(loc = 'upper left')
    mpl.show()


#Tempreature class 
class TempPredictor:
    
    #The (scaled) data frame
    D = None
    #Unscaled timestamp data
    DTS = None
    #The data matrix
    A = None
    #Target value matrix
    y = None
    #Corresponding columns for target values
    targCols = None
    #Number of previous days of data to use
    npd = 1
    #The regressor model
    R = None
    #Object to scale input data
    S = None
    
    #Constructor
    #nPrevDays:     The number of past days to include
    #               in a sample.
    #rmodel:        The regressor model to use (sklearn)
    #nPastDays:     The number of past days in each feature
    #scaler:        The scaler object used to scale the data (sklearn)
    def __init__(self, rmodel, nPastDays , scaler = StandardScaler()):
        self.npd = nPastDays
        self.R = rmodel
        self.S = scaler
        
    #Extracts features from data
    #
    #D:         A dataframe from ParseData
    #ret:       The data matrix of samples
    def _ExtractFeat(self, D):
        #One row per day of stock data
        m = D.shape[0]
        #print(m)
        #Open, High, Low, and Close for past n days + timestamp and volume
        n = self._GetNumFeatures()
        B = np.zeros([m, n])
        #Preserve order of spreadsheet
        for i in range(m - 1, -1, -1):
            self._GetSample(B[i], i, D)
        #Return the internal numpy array
        return B
        
    #Extracts the target values from data
    #
    #D:         A dataframe from ParseData
    #ret:       The data matrix of targets and the
    
    def _ExtractTarg(self, D):
        #Timestamp column is not predicted
        tmp = D.drop('Timestamp', axis = 1)
        #Return the internal numpy array
        print(tmp.columns)
        return tmp.values, tmp.columns
        
    #Get the number of features in the data matrix
    #
    #n:         The number of previous days to include
    #           self.npd is  used if n is None
    #ret:       The number of features in the data matrix
    def _GetNumFeatures(self, n = None):
        if(n is None):
            n = self.npd
        return n * 7 + 1
        
    #Get the sample for a specific row in the dataframe. 
    #A sample consists of the current timestamp and the data from
    #the past n rows of the dataframe
    #r:         The array to fill with data
    #i:         The index of the row for which to build a sample
    #df:        The dataframe to use
    #return;    r
    def _GetSample(self, r, i, df):
        #First value is the timestamp
        r[0] = df['Timestamp'].values[i]
        #The number of columns in df
        n = df.shape[1]
        #The last valid index
        lim = df.shape[0]
        #Each sample contains the past n days of data; for non-existing data
        #repeat last available sample
        #Format of row:
        #Timestamp Volume Open[i] High[i] ... Open[i-1] High[i-1]... etc
        for j in range(0, self.npd):
            #Subsequent rows contain older data in the spreadsheet
            ind = i + j + 1
            #If there is no older data, duplicate the oldest available values
            if(ind >= lim):
                ind = lim - 1
            #Add all columns from row[ind]
            for k, c in enumerate(df.columns):
                #+ 1 is needed as timestamp is at index 0
                r[k + 1 + n * j] = df[c].values[ind]
        return r
        
    #Attempts to learn the stock market data
    #given a dataframe taken from ParseData
    #
    #D:         A dataframe from ParseData
    def Learn(self, D):
        #Keep track of the currently learned data
        self.D = D.copy()
        #Keep track of old timestamps for indexing
        self.DTS = np.copy(D.Timestamp.values)
        #Scale the data
        self.D[self.D.columns] = self.S.fit_transform(self.D)
        #Get features from the data frame
        self.A = self._ExtractFeat(self.D)
        #Get the target values and their corresponding column names
        self.y, self.targCols = self._ExtractTarg(self.D)
        #Create the regressor model and fit it
        self.R.fit(self.A, self.y)

        
    #Predicts values for each row of the dataframe. Can be used to 
    #estimate performance of the model
    #
    #df:            The dataframe for which to make prediction
    #return:        A dataframe containing the predictions
    def PredictDF(self, df):
        #Make a local copy to prevent modifying df
        D = df.copy()
        #Scale the input data like the training data
        D[D.columns] = self.S.transform()
        #Get features
        A = self._ExtractFeat(D)
        #Construct a dataframe to contain the predictions
        #Column order was saved earlier 
        P = pd.DataFrame(index = range(A.shape[0]), columns = self.targCols)
        #Perform prediction 
        P[P.columns] = self.R.predict(A)
        #Add the timestamp (already scaled from above)
        P['Timestamp'] = D['Timestamp'].values
        #Scale the data back to original range
        P[P.columns] = self.S.inverse_transform(P)
        return P
        
    #Predict the stock price during a specified time
    #
    #startDate:     The start date as a string in yyyy-mm-dd format
    #endDate:       The end date as a string yyyy-mm-dd format
    #period:		'daily', 'weekly', or 'monthly' for the time period
    #				between predictions
    #return:        A dataframe containing the predictions or
    def PredictDate(self, startDate, endDate, period = 'daily'):
        #Create the range of timestamps and reverse them
        ts = DateRange(startDate, endDate, period)[::-1]
        m = ts.shape[0]
       
        #Prediction is based on data prior to start date
        #Get timestamp of previous day
        prevts = DatePrevDay(ts[-1])
        #print (prevts)
        #Test if there is enough data to continue
        try:
            ind = np.where(self.DTS == prevts)[0][0]
        except IndexError:
            return None
        #There is enough data to perform prediction; allocate new data frame
        P = pd.DataFrame(np.zeros([m, self.D.shape[1]]), index = range(m), columns = self.D.columns)
        #Add in the timestamp column so that it can be scaled properly
        P['Timestamp'] = ts
        #Scale the timestamp (other fields are 0)
        P[P.columns] = self.S.transform(P)
        #B is to be the data matrix of features
        B = np.zeros([1, self._GetNumFeatures()])
        #Add extra last entries for past existing data
        for i in range(self.npd):
            #If the current index does not exist, repeat the last valid data
            curInd = ind + i
            if(curInd >= self.D.shape[0]):
                curInd = curInd - 1
            #Copy over the past data (already scaled)
            P.loc[m + i] = self.D.loc[curInd]
        #Loop until end date is reached
        for i in range(m - 1, -1, -1):
            #Create one sample 
            self._GetSample(B[0], i, P)
            #Predict the row of the dataframe and save it
            pred = self.R.predict(B).ravel()
            #Fill in the remaining fields into the respective columns
            for j, k in zip(self.targCols, pred):
                P.set_value(i, j, k)
        #Discard extra rows needed for prediction
        P = P[0:m]
        #Scale the dataframe back to the original range
        P[P.columns] = self.S.inverse_transform(P)
        return P
        
    #Test the predictors performance and
    #displays results to the screen
    #
    #D:             The dataframe for which to make prediction
    def TestPerformance(self, df = None):
        #If no dataframe is provided, use the currently learned one
        if(df is None):
            D = self.D
        else:
            D = self.S.transform(df.copy())
        #Get features from the data frame
        A = self._ExtractFeat(D)
        #Get the target values and their corresponding column names
        y, _ = self._ExtractTarg(D)
        #Begin cross validation
        kf = KFold(A.shape[0])
        for trn, tst in kf:
            s1 = self.R.score(A, y)
            s2 = self.R.score(A[tst], y[tst])
            s3 = self.R.score(A[trn], y[trn])
            print('C-V:\t' + str(s1) + '\nTst:\t' + str(s2) + '\nTrn:\t' + str(s3))


def PrintUsage():
    print('Usage:\n')
    print('\tpython stocks.py <csv file> <start date> <end date> <D|W|M>')
    print('\tD: Daily prediction')
    print('\tD: Weekly prediction')
    print('\tD: Montly prediction')

#Main Method 
def Main(args):
    if(len(args) != 3 and len(args) != 4 and len(args)!= 5):
        PrintUsage()
        return
    #Test if file exists
    try:
        open(args[0])
    except Exception as e:
        print('Error opening file: ' + args[0])
        print(str(e))
        PrintUsage()
        return
    #Test validity of start date string
    try:
        datetime.strptime(args[1], '%Y-%m-%d').timestamp()
    except Exception as e:
        print('Error parsing date: ' + args[1])
        PrintUsage()
        return
    #Test validity of end date string
    try:
        datetime.strptime(args[2], '%Y-%m-%d').timestamp()
    except Exception as e:
        print('Error parsing date: ' + args[2])
        PrintUsage()
        return    
    #Test validity of final optional argument
    if(len(args) == 4):
        predPrd = args[3].upper()
        if(predPrd == 'D'):
            predPrd = 'daily'
        elif(predPrd == 'W'):
            predPrd = 'weekly'
        elif(predPrd == 'M'):
            predPrd = 'monthly'
        else:
            PrintUsage()
            return
    else:
        predPrd = 'daily'
    #Everything looks okay; proceed with program
    #Grab the data frame
    D = ParseData(args[0])
    #The number of previous days of data used
    #when making a prediction
    numPastDays = 10
    PlotData(D)
    #Number of neurons in the input layer
    i = numPastDays * 7 + 1
    #Number of neurons in the output layer
    o = D.shape[1] - 1
    #Number of neurons in the hidden layers
    h = int((i + o) / 2)
    #The list of layer sizes
    layers = [i, h, h, h, h, h, h, o]
    R = MLPR(layers, maxItr = 1000, tol = 0.40, reg = 0.001, verbose = True)
    sp = TempPredictor(R, nPastDays = numPastDays)
    #Learn the dataset and then display performance statistics
    sp.Learn(D)
    sp.TestPerformance()
    #Perform prediction for a specified date range
    P = sp.PredictDate(args[1], args[2], predPrd)
    #Keep track of number of predicted results for plot
    n = P.shape[0]
  
    length = P['Temp'].values
    temp = np.zeros(length.shape)
    acttemp = np.zeros(length.shape)
    predtemp = np.zeros(length.shape)

    
    for i  in range(len(temp)):
	    temp[i] = D.iloc[i]['Temp'] - P.iloc[i]['Temp']
	    acttemp[i]= D.iloc[i]['Temp'] 
	    predtemp[i] = P.iloc[i]['Temp']
	   

    #create csv
    Header = ["Actual","Predicted","Error"]
    df = pd.DataFrame({'Actual':acttemp,'Predicted':predtemp,'Error':temp})
    df.to_csv(args[4],columns = Header)
    
   
    #Append the predicted results to the actual results
    D = P.append(D)
    #Predicted results are the first n rows
    PlotData(D, range(n + 1))   
    return (P, n)


if __name__ == "__main__":
    Main(sys.argv[1:])
    #p, n = Main(['/home/mai/Desktop/trial5/yahoostockorg.csv', '2016-02-20', '2016-02-28', 'D'])
