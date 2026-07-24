import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv("Car_Dekho_Cleaned.csv")

# Features & target
X = df.drop(columns='Price')
y = df['Price']

# One Hot Encoding
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name','company','fuel_type']),
    remainder="passthrough"
)

# Train 1000 models
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )
    
    pipe = make_pipeline(column_trans, LinearRegression())
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

scores = np.array(scores)

# Best model
best_index = np.argmax(scores)
best_score = scores[best_index]

print("Best Model Index:", best_index)
print("Best R2 Score:", best_score)

# Plot graph
plt.figure()
plt.plot(scores)
plt.scatter(best_index, best_score)

plt.title("R2 Score vs Model Number")
plt.xlabel("Model Number")
plt.ylabel("R2 Score")

plt.annotate(
    f'Best: {best_index}\nScore: {best_score:.3f}',
    (best_index, best_score),
    textcoords="offset points",
    xytext=(10,10)
)

plt.show()