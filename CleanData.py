import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras
import locale

locale.setlocale(locale.LC_ALL, "hu_HU")

pd.set_option('display.html.table_schema', True)  # to can see the dataframe/table as a html
pd.set_option('display.precision', 5)

rawPath = "C:/DEV/hadoop/data.csv"

rawData = pd.read_csv(rawPath, sep=';', header=0,
                      dtype={
                          "CurrentTime": object, "RouteID": object, "TripID": object,
                          "ArrivalDiff": int, "DepartureDiff": int, "Temperature": float,
                          "Humidity": int, "Preasure": float, "WindIntensity": float,
                          "SnowIntesity": float, "RainIntensity": float
                      }
                      )
rawData.replace(',', '.', regex=True, inplace=True)
print("Beolvasás kész")

cleanData = rawData.drop(
    columns=['VeichleID', 'VeichleType', 'VeichleModel', 'VeichleLongitude', 'VeichleLatitude', 'StopID',
             'StopSequance', 'PredictedDepartureTime', 'DepartureTime', 'TripStatus', 'ArrivalTime',
             'PredictedArrivalTime'])
print("Feleseges cellák törlve")

cleanData['CurrentTime'] = pd.to_datetime(cleanData['CurrentTime'], unit='ms')
cleanData['CurrentTime'] = cleanData['CurrentTime'].dt.hour
cleanData['Temperature'] = pd.to_numeric(cleanData['Temperature'])
cleanData['WindIntensity'] = pd.to_numeric(cleanData['WindIntensity'])
cleanData['RainIntensity'] = pd.to_numeric(cleanData['RainIntensity'])
cleanData['SnowIntesity'] = pd.to_numeric(cleanData['SnowIntesity'])
cleanData.head(10)

aggregatedData = cleanData.groupby(['CurrentTime', 'RouteID', 'TripID']).agg({
    'ArrivalDiff': 'mean',
    'DepartureDiff': 'mean',
    'Temperature': 'mean',
    'WindIntensity': 'mean',
    'RainIntensity': 'mean',
    'SnowIntesity': 'mean',
})

aggregatedData['result'] = aggregatedData[['ArrivalDiff', 'DepartureDiff']].mean(axis=1)
aggregatedData['result'] = aggregatedData['result'].apply(lambda x: -1 if x >= 4000 else x)

mask = aggregatedData['result'] != -1
aggregatedData = aggregatedData[mask]

aggregatedData = aggregatedData.drop(columns=['ArrivalDiff', 'DepartureDiff'])
aggregatedData.to_csv("C:/DEV/hadoop/clean_data.csv", sep=';', header=True, float_format='%.15f')

print(aggregatedData['result'].min())
print(aggregatedData['result'].max())

aggregatedData.head(10)

normalizeddData = aggregatedData.copy(deep=True)
standardizedData = aggregatedData.copy(deep=True)

data = normalizeddData[['result']].values
normalizeddData[['result']] = keras.utils.normalize(
    data,
    axis=0,
    order=10
)
normalizeddData.to_csv("C:/DEV/hadoop/normalized_data.csv", sep=';', header=True, float_format='%.15f')
normalizeddData[['result']].head(10)
print(normalizeddData['result'].min())
print(normalizeddData['result'].max())
normalizeddData['result'].plot(kind='kde')

scaler = StandardScaler()
scaler = scaler.fit(standardizedData[['result']])

normalized = scaler.transform(standardizedData[['result']])
standardizedData['result'] = normalized
standardizedData['result'].plot(kind='kde')
print(standardizedData['result'].min())
print(standardizedData['result'].max())

normalizeddData.to_csv("C:/DEV/hadoop/standardized_data.csv", sep=';', header=True, float_format='%.15f')
