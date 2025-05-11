import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("C:/MI/data/StudentsPerformance.csv")

print("Oszlopok:", df.columns.tolist())

label_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['math score'])  # bemeneti változók
y = df['math score']                 # célváltozó

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ellenőrzésképp:
print("X_train alak:", X_train.shape)
print("y_train minta:", y_train.head())

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel teljesítménye:")
print(f"Átlagos négyzetes hiba (MSE): {mse:.2f}")
print(f"Determinációs együttható (R²): {r2:.2f}")

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Valós pontszám")
plt.ylabel("Előrejelzett pontszám")
plt.title("Valós vs Előrejelzett pontszámok")
plt.show()