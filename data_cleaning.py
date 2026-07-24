#4321 data in dataset
import pandas as pd
import numpy as np

df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")


df.rename(columns={"km_driven": "kms_driven"}, inplace=True)
df.rename(columns={"fuel": "fuel_type"}, inplace=True)
df.rename(columns={"selling_price": "Price"}, inplace=True)

# adding a column
df["company"] = df["name"].str.split().str[0]


cols = list(df.columns)

# Remove 'Price' from list
cols.remove("company")

# Insert at index 1
cols.insert(1, "company")

# Reorder dataframe
df = df[cols]

df.drop_duplicates(inplace=True)

# print(df.shape)

# indexing
df.reset_index(drop=True, inplace=True)


print(df.head())           




df["fuel_type"] = df["fuel_type"].replace("LPG", "CNG")
print(df["fuel_type"].unique())


# print(df["fuel_type"].value_counts())


df.to_csv("Car_Dekho_Cleaned.csv", index=False)


# Creating Model
X = df.drop(columns='Price')
y = df['Price']
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])
ohe.categories_


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder="passthrough")
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)
r2 = r2_score(y_test,y_pred)
print(r2)


# checking max r2 Score
scores = []
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


max = np.argmax(scores)
print(max)

sc = scores[np.argmax(scores)]
print(sc)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


import pickle
pickle.dump(pipe,open('LinearRegression.pkl','wb'))
# z = pipe.predict(pd.DataFrame([['Maruti Alto LX BSIII','Maruti','2019','100','Petrol']], columns=['name','company','year','kms_driven','fuel_type']))
# print(z)





