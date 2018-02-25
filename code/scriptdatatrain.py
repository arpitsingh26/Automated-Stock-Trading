import csv
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import linear_model
from numpy import nan
from sklearn import svm
from itertools import izip

#csize indicates the number of days which is merged
csize=10

#parameters that will be taken as a part of training data
priceopen=[]
priceclose=[]
pricelow=[]
pricehigh=[]
volumeopen=[]
volumeclose=[]
volumelow=[]
volumehigh=[]
startdate=[]
enddate=[]

#taking the required parameters as a part of training data from the given file
with open('tatatraingiven.csv', 'r') as csvfile:
	csvData = csv.reader(csvfile, delimiter=',')
	csvData.next()
	i=0
	temppricelow=2147483647
	temppricehigh=0
	tempvolumelow=2147483647
	tempvolumehigh=0
	arraysize=0
	for row in csvData:
		if i==0:
			priceclose.append(float(row[4])) #priceclose for the interval is priceclose for the last day of the interval
			volumeclose.append(int(row[6])) #volumeclose for the interval is volume for the last day of the interval
			enddate.append(str(row[0]))

		temppricelow=min(temppricelow,float(row[3]))
		temppricehigh=max(temppricehigh,float(row[2]))
		tempvolumelow=min(tempvolumelow,int(row[6]))
		tempvolumehigh=max(tempvolumehigh,int(row[6]))

		i=i+1

		if i==csize:
			pricehigh.append(float(temppricehigh)) #pricehigh for the interval is the highest price during the interval
			temppricehigh=0
			pricelow.append(float(temppricelow)) #pricelow for the interval is the lowest price during the interval
			temppricelow=2147483647
			priceopen.append(float(row[1])) #priceopen for the interval is priceopen for the first day of the interval
			volumehigh.append(int(tempvolumehigh)) #volumehigh for the interval is the highest volume during the interval
			tempvolumehigh=0
			volumelow.append(int(tempvolumelow)) #volumelow for the interval is the lowest volume during the interval
			tempvolumelow=2147483647
			volumeopen.append(int(row[6])) #volumeopen for the interval is volume for the first day of the interval
			startdate.append(str(row[0]))
			arraysize=arraysize+1
			i=0

#writing data into training file
with open('tatatrainactual.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Start date','End date','Open Price','Close Price','Low Price','High Price','Open Volume','Close Volume','Low Volume','High Volume',])
    for i1 in range(arraysize):
    	i=arraysize-1-i1
    	writer.writerow((startdate[i],enddate[i],priceopen[i],priceclose[i],pricelow[i],pricehigh[i],volumeopen[i],volumeclose[i],volumelow[i],volumehigh[i]))

