# =================== CAR PRICE PREDICTION ===================
import pandas as pd

df = pd.read_csv(
    r"C:\Users\Akansha Singh\Downloads\car data (1).csv"
)

print(df.head(5))

df.info()

df2 = df.select_dtypes(['int', 
'float'])
print(df2)

print(df2.corr())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Fuel_Type'] = encoder.fit_transform(df.Fuel_Type)

df2 = df.drop_duplicates()

print(df2.isnull().sum())

X = df2[['Year', 'Present_Price', 'Fuel_Type', 'Kms_Driven']]
y = df2['Selling_Price']

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)

predicted = regr.predict([[2020, 9.85, 2, 8000]])
print(predicted)

predicted2 = regr.predict(X)
print(predicted2)

from sklearn.metrics import r2_score
print(r2_score(y, predicted2))