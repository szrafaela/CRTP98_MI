import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ellenőrizzük a train.csv fájlt
data_file = 'train.csv'
if not os.path.exists(data_file):
    print(f"Hiba: A '{data_file}' fájl nem található!")
    print("Töltsd le a Titanic adathalmazt: https://www.kaggle.com/competitions/titanic/data")
    print("Helyezd a 'train.csv' fájlt ebbe a mappába.")
    exit(1)

try:
    # Adatok betöltése
    data = pd.read_csv(data_file)
    print("Adathalmaz sikeresen betöltve!")
    print("\nElső 5 sor:")
    print(data.head())

    # Egyszerű előfeldolgozás
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    features = ['Pclass', 'Sex', 'Age']
    X = data[features]
    y = data['Survived']

    # Adatok felosztása
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell tanítása
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Előrejelzés és pontosság
    y_pred = model.predict(X_test)
    print("\nModell pontossága:", accuracy_score(y_test, y_pred))

    # Vizualizáció
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Sex', y='Survived', data=data)
    plt.title('Túlélési arány nemek szerint')
    plt.xlabel('Nem (0 = férfi, 1 = nő)')
    plt.ylabel('Túlélési arány')
    plt.savefig('survival_by_sex.png')
    print("Grafikon mentve: survival_by_sex.png")
    plt.show()

except Exception as e:
    print(f"Hiba történt: {str(e)}")
    print("Ellenőrizd, hogy a könyvtárak (pandas, scikit-learn, matplotlib, seaborn) telepítve vannak.")