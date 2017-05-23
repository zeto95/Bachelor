# Bachelor

#There are five files in this folder

#(openhab.csv):The actual dataset

#(actual7):The transformed dataset

#(actual.py): Which contains the main code of the model 

#(datatransformation.py): included the code for the data transformation in a separate file, simply i took the average tempreature of each day to be feed into the model. 

#(TFMLP.py):This file contains a class implementing a multi-layer perceptron (MLP) using numpy and tensorflow. 

#To run the example, the TFMLP file should run first and then the actual.py in the terminal. 

#The inputs for the command line is as follows:

#startdate: of the prediction (year/month/day) ex:2017-02-16

#enddate: of the prediction (year/month/day) ex:2017-03-20

#D/M/W: how to present the predicted values(daily/monthly/weekly)

#<The path of the actual7.py file> 'startdate' 'enddate' 'D' 'The path of the results -csv file- to be'

#example of command line
python3 actual.py '/home/mai/Desktop/actualtest/actual7.csv' '2017-02-09' '2017-04-20' 'D' '/home/mai/Desktop/Outputs/newdata.csv'

