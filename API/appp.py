from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error ,  r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor

data= fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
print(df.head)
df['Price']=data.target
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mae)
r2= r2_score(y_test, y_pred)
print("R2 Score:", r2)
print(f"average score: ${mae*100000:,.0f}")
joblib.dump(model,'house_model.joblib')
joblib.dump(list(X.columns),'house_features.joblib')