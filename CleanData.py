import pandas as pd
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

aggregatedData = cleanData.groupby(['CurrentTime', 'RouteID', 'TripID'], as_index=False).agg({
    'ArrivalDiff': 'mean',
    'DepartureDiff': 'mean',
    'Temperature': 'mean',
    'WindIntensity': 'mean',
    'RainIntensity': 'mean',
    'SnowIntesity': 'mean',
})

aggregatedData['result'] = aggregatedData[['ArrivalDiff', 'DepartureDiff']].mean(axis=1)
aggregatedData = aggregatedData.drop(columns=['ArrivalDiff', 'DepartureDiff', 'TripID'])
aggregatedData = aggregatedData[aggregatedData['result'] < 3500]
aggregatedData = aggregatedData[aggregatedData['result'] > 60]

aggregatedData['RouteID'] = aggregatedData['RouteID'].apply(lambda row: pd.to_numeric(row.split('_')[1]))

print("A megtalálható legnagybb érték most: {0}".format( aggregatedData['result'].max()))

aggregatedData.to_csv("C:/DEV/hadoop/clean_data.csv", sep=';', header=True, float_format='%.15f')

print(aggregatedData['result'].min())
print(aggregatedData['result'].max())

aggregatedData.head(10)