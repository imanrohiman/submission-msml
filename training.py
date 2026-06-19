
from preprocessing.preprocessing import load_data
from modeling.modeling import train_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = load_data("dataset/titanic.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = train_model(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "membangun_model/model/model.pkl")
