import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

penguins = pd.read_csv(url).dropna(subset=features + ["species"])
X = penguins[features]
y = penguins["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test accuracy: {accuracy:.1%} ({len(y_test)} penguins)")
