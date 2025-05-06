# Szükséges könyvtárak importálása
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Adathalmaz betöltése
train_data = pd.read_csv('train.csv')

# 2. Adat-előkészítés
# Hiányzó 'Age' értékek kitöltése az átlagos életkorral
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

# Hiányzó 'Embarked' értékek kitöltése a leggyakoribb értékkel
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# 'Cabin' oszlop elhagyása, mert túl sok a hiányzó érték
train_data = train_data.drop('Cabin', axis=1)

# Kategorikus változók számmá alakítása (pl. 'Sex' és 'Embarked')
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)

# 3. Adathalmaz felosztása 
# Célváltozó (Survived) és jellemzők kiválasztása
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = train_data['Survived']

# Tanító és tesztelő adatokra bontás
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ellenőrzés: Nézzük meg az adatok méretét
print("Tanító adatok mérete:", X_train.shape)
print("Tesztelő adatok mérete:", X_test.shape)

# 4. Modell betanítása 
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Predikció és kiértékelés 
y_pred = model.predict(X_test)
print("Pontosság:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. Konfúziós mátrix vizualizáció
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Konfúziós mátrix')
plt.xlabel('Predikált')
plt.ylabel('Tényleges')
plt.savefig('confusion_matrix.png')
plt.show()