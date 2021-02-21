#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt      
        
data = pd.read_csv(r'C:\...\AirPassengers.csv')


# In[3]:


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv(r'C:\...\AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
print (data)


# In[4]:


ts = data['#Passengers']
ts['1949-01-01']

ts['1957-01-01':'1957-12-01']


# In[5]:


plt.plot(ts)


# In[9]:


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)


# In[10]:


test_stationarity(ts)


# In[11]:


ts_log = np.log(ts)
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[12]:


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# In[15]:


expwighted_avg = ts_log.ewm(12).mean()
log = plt.plot(ts_log, label='Log')
exp = plt.plot(expwighted_avg, color='red', label='Log expwighted_avg')
avg = plt.plot(moving_avg, color='green', label='Log moving_avg')
plt.legend(loc='best')
plt.title('Log - expwighted_avg - moving_avg')
plt.show(block=False)


# In[16]:


ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)


# In[24]:


from statsmodels.tsa.arima_model import ARIMA       


model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)


predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(ts)
plt.plot(predictions_ARIMA)

plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))


# In[ ]:




