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