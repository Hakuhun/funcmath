import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB

pd.set_option('display.html.table_schema', True) # to can see the dataframe/table as a html
pd.set_option('display.precision', 5) 

rawPath = "C:/DEV/hadoop/normalized_data.csv"
data = pd.read_csv(rawPath, sep=';', header=0)
print("Beolvasás kész")
data = data.drop(columns=['TripID', 'RouteID', 'CurrentTime'], axis = 1)

print('Minta')
print(data['result'].min())
print(data['result'].max())

print('ELoszlás ')
print('Dispersion of lates, normalied the maximum to 3600 seconds (1 hour)')
data['result'].plot.kde(bw_method=0.3)
data.head(10)



#data['RouteID'] = data['RouteID'].apply(lambda row: pd.to_numeric(row.split('_')[1]))
data['result'] = data['result'].apply(lambda row: -1 if row > 3600 else row)
x = data.iloc[:,0:3]
y = data.iloc[:,3:4]

data['result'].plot.kde(bw_method=0.1)
print(x.shape)
print(y.shape)

model = MultinomialNB()

# Train the model using the training sets 
model.fit(x, np.ravel(y.values))

prediction = model.predict_proba(data.iloc[0:1, 0:3])
print (prediction)
