import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('application_train.csv', nrows=50000)

target = 'TARGET'
X = df.drop(columns=[target])
y = df[target]

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

for col in X.select_dtypes(include='number').columns:
    X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 5 Most Important Features:")
print(feature_importance.nlargest(5))
