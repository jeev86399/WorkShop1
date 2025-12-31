import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv('car data.csv')
df.head()
#drop car_name column
df=df.drop('Car_Name',axis=1)
df.head()
#convert categorical to numerical
df_encoded=pd.get_dummies(df,drop_first=True)
df_encoded.head()
X=df_encoded.drop('Selling_Price',axis=1)
y=df_encoded['Selling_Price']
#train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#train ml model
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(
    n_estimators=200,
    criterion='squared_error',
    random_state=42
)
model.fit(X_train,y_train)
#evaluate model
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
y_pred=model.predict(X_test)
print("Mean aquared error",r2_score(y_test,y_pred))
print("r2_score",mean_squared_error(y_test,y_pred))
print("mean absolute error",mean_absolute_error(y_test,y_pred))
def predict_price(year,
                  present_price,
                  km_driven,
                  fuel_type,
                  owner,
                  company,
                  seller_type,
                  transmission):

    input_data = pd.DataFrame({
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [km_driven],
        'Fuel_Type_' + fuel_type: [True],
        'Owner': [owner],
        'Company_' + company: [True],
        'seller_type_' + seller_type: [True],
        'Transmission_' + transmission: [True]
    })

    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

    predicted_price = model.predict(input_data_encoded)[0]
    return predicted_price
price = predict_price(
    year=2019,
    present_price=10.59,
    km_driven=2700,
    fuel_type='Petrol',
    owner=1,
    company='toyota',
    seller_type='Individual',
    transmission='Manual'
)
print("Predicted Price:", price)
