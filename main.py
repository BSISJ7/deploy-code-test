from math import sqrt
import pandas as pandasFuncs
import numpy as numpyFuncs
import os
import sklearn.linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import json
import requests as apiRequests
import joblib




global TEMPERATURE_DATA_CSV
TEMPERATURE_DATA_CSV = os.getcwd()+'/TempData/Weather in Szeged 2006-2016 (copy).csv'


def main():
    
# ========================================== Collect Weather Data ========================================== #
    
    
    
    API_KEY = '0b5c51277ca90405ac816ebeb8d4b9c6'
    BASE_URL = "https://api.openweathermap.org/data/2.5/onecall?lat=33.44&lon=-94.04&exclude=daily,minutely,current&" \
        "units=metric&appid="+API_KEY
    
    
    
    weatherData = apiRequests.get(BASE_URL).json()
    dataDict = {'Temperature (C)':[], 'Humidity':[], 'Wind Speed (km/h)':[], 'Wind Bearing (degrees)':[],
               'Visibility (km)':[], 'Pressure (millibars)':[]}
    
    for x in range(0, len(weatherData['hourly'])):
        dataDict['Temperature (C)'].append(weatherData['hourly'][x]['temp'])
        dataDict['Humidity'].append(weatherData['hourly'][x]['humidity'])
        dataDict['Wind Speed (km/h)'].append(weatherData['hourly'][x]['wind_speed'])
        dataDict['Wind Bearing (degrees)'].append(weatherData['hourly'][x]['wind_deg'])
        dataDict['Visibility (km)'].append(weatherData['hourly'][x]['visibility']/1000)
        dataDict['Pressure (millibars)'].append(weatherData['hourly'][x]['pressure'])

    panDataFrame = pandasFuncs.DataFrame(dataDict)
    
    
# ========================================== Modify Pressure ========================================== #
    
    
    Q1 = panDataFrame["Pressure (millibars)"].quantile(0.10)
    Q3 = panDataFrame["Pressure (millibars)"].quantile(0.95)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    panDataFrame['Pressure (millibars)'] = \
        numpyFuncs.where(panDataFrame['Pressure (millibars)'] > \
                         upper_limit, upper_limit, panDataFrame['Pressure (millibars)'])
    panDataFrame['Pressure (millibars)'] = \
        numpyFuncs.where(panDataFrame['Pressure (millibars)'] < \
                         lower_limit, lower_limit, panDataFrame['Pressure (millibars)'])
    
    
# ========================================== Transform Weather Data ========================================== #
   
    
    sqrt_transformer = FunctionTransformer(numpyFuncs.sqrt)
    transformedWindSpd = sqrt_transformer.transform(panDataFrame['Wind Speed (km/h)'])
    panDataFrame['Wind Speed (km/h)'] = transformedWindSpd

    expTransformer = FunctionTransformer(numpyFuncs.exp)
    transformedHumidity = expTransformer.transform(panDataFrame['Humidity'])
    panDataFrame['Humidity'] = transformedHumidity

    expTransformer = FunctionTransformer(numpyFuncs.exp)
    transformedVisibility = expTransformer.transform(panDataFrame['Visibility (km)'])
    panDataFrame['Visibility (km)'] = transformedVisibility

    
# ========================================== Wind Bearing Data Scaling ========================================== #
    
    
    discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
    windTrainingData = pandasFuncs.DataFrame(panDataFrame, columns=['Wind Bearing (degrees)'])
    discretizer.fit(windTrainingData)
    discWindTrainData = pandasFuncs.DataFrame(discretizer.transform(windTrainingData))
    panDataFrame['Wind Bearing (degrees)'] = discWindTrainData
    

# ========================================== Data Standardization ========================================== #

   
    dataColumnNames = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)',
                          'Visibility (km)', 'Pressure (millibars)']
    standardization = StandardScaler()
    standardization.fit(panDataFrame[dataColumnNames])
    transformedDataScaled = standardization.transform(panDataFrame[dataColumnNames])
    panDataFrame[dataColumnNames] = transformedDataScaled
    
    
# ========================================== Load Model And Predict Outcome ========================================== #
    
    joblib_file = "joblib_RL_Model.pkl"  
    joblib_LR_model = joblib.load(joblib_file)
#     print(joblib_LR_model.predict(panDataFrame))
    
    
# ========================================== Display Data ========================================== #
 
    
    dataDict = {'Date':[], 'Temperature':[]}

    for x in range(0, len(weatherData['hourly'])):
        date = datetime.datetime.fromtimestamp(float(weatherData['hourly'][x]['dt']))
        dataDict['Date'].append(date)
        dataDict['Temperature'].append(weatherData['hourly'][x]['temp'])


    x = dataDict['Date']
    y = dataDict['Temperature']

    plt.figure(figsize=(14,8))
    plt.title('Hourly Temperature Readings', fontweight ="bold")
    plt.xlabel('Date and Hour')
    plt.ylabel('Apparent Temperature (C)')
    plt.plot_date(dataDict['Date'], dataDict['Temperature'])
    plt.show()

    print('\n\n\n')

    ax = plt.axes()
    d_data = pd.DataFrame.from_dict(weatherData['hourly'])
    d_data.head(10)
    sns.set(rc={"figure.figsize":(14, 8)})
    sns.heatmap(d_data.corr(),ax=ax,annot=True)
    ax.set_title('Weather Data Heatmap')
    
main()
print('\n\nDone')
