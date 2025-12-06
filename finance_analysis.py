import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay


df = yf.download("GOOGL",start="2025-01-1" ,end="2025-12-5")
df.columns = ["_".join(col).strip() for col in df.columns.values]

df["Z_score"] = (df["Close_GOOGL"]- df["Close_GOOGL"].mean())/df["Close_GOOGL"].std()

def risk_rule(z):
    if z > 1:
        return "High Risk"
    elif z >= -1 and z <= 1:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Level"] = df["Z_score"].apply(risk_rule)

google = df.reset_index()
mapping = {
    "Low Risk": 0,
    "Medium Risk": 1,
    "High Risk": 2
}
google["Risk_Lavel"]= google["Risk_Level"].map(mapping)
google["Risk_Lavel"] = google["Risk_Lavel"].shift(-1)
google = google.dropna()

print(google.head(50))

x = google[["Close_GOOGL","High_GOOGL","Low_GOOGL","Open_GOOGL","Volume_GOOGL","Z_score"]]
y= google["Risk_Lavel"]

x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# we r useing knn model
print("knn model result")
print(50*"_")
print("Here Low_risk = 0\n Medium_risk = 1\n High_risk = 2 ")
# findind best value 
k_values = range(1, 31)  
accuracy_scores = []
for k in k_values:
    pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_test)
    accuracy_scores.append(accuracy_score(y_test, preds))
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print(f"Best k value is {best_k}")
model_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k))
])
model_knn.fit(x_train,y_train)

model_knn_predict = model_knn.predict(x_test)

print(f"Knn model prediction is {model_knn_predict}")
acc_score_knn = accuracy_score(y_test,model_knn_predict)
print(f" Accurecy score of this model is {acc_score_knn}")
cm1 = confusion_matrix(y_test, model_knn_predict)
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=["Low","Medium","High"]).plot(ax=ax)
ax.set_title("KNN Confusion Matrix")
plt.show()
print(50*"_")

print("Decision Tree model result")
print(50*"_")
print("Here Low_risk = 0\n Medium_risk = 1\n High_risk = 2 ")

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(x_train,y_train)

model_dt_predict = model_dt.predict(x_test)
print(f"Decision tree model prediction is {model_dt_predict}")
acc_score_dt = accuracy_score(y_test,model_dt_predict)
print(f"The accurecy score of Decision tree model is {acc_score_dt}")
cm2 = confusion_matrix(y_test, model_dt_predict)
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Low","Medium","High"]).plot(ax=ax)
ax.set_title("Decision Tree Confusion Matrix")
plt.show()
print(50*"_")

print("Random Forest model result")
print(50*"_")
print("Here Low_risk = 0\n Medium_risk = 1\n High_risk = 2 ")

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(x_train,y_train)

model_rf_predict = model_rf.predict(x_test)
print(f"Random Forest model prediction is {model_rf_predict}")

acc_score_rf = accuracy_score(y_test, model_rf_predict)
print(f"The accuracy score of Random Forest model is {acc_score_rf}")
cm3 = confusion_matrix(y_test, model_rf_predict)
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=["Low","Medium","High"]).plot(ax=ax)
ax.set_title("Random Forest Confusion Matrix")
plt.show()
print(50*"_")

print("SVM model result kernel = rbf")
print(50*"_")
print("Here Low_risk = 0\n Medium_risk = 1\n High_risk = 2 ")

model_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf'))  
])

model_svm.fit(x_train, y_train)

model_svm_predict = model_svm.predict(x_test)
print(f"SVM model prediction is {model_svm_predict}")

acc_score_svm = accuracy_score(y_test, model_svm_predict)
print(f"The accuracy score of SVM model is {acc_score_svm}")
cm4 = confusion_matrix(y_test, model_svm_predict)
fig, ax = plt.subplots(figsize=(5,4))
ConfusionMatrixDisplay(confusion_matrix=cm4, display_labels=["Low","Medium","High"]).plot(ax=ax)
ax.set_title("SVM (rbf) Confusion Matrix")
plt.show()
print(50*"_")


print("SVM model result kernel = sigmoid")
print(50*"_")
print("Here Low_risk = 0\n Medium_risk = 1\n High_risk = 2 ")

model_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='sigmoid'))  
])

model_svm.fit(x_train, y_train)

model_svm_predict = model_svm.predict(x_test)
print(f"SVM model prediction is {model_svm_predict}")

acc_score_svm = accuracy_score(y_test, model_svm_predict)
print(f"The accuracy score of SVM model is {acc_score_svm}")
print(50*"_")

print("SVM model result kernel = poly")
print(50*"_")
print("Here Low_risk = 0\n Medium_risk = 1\n High_risk = 2 ")

model_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='poly'))  
])

model_svm.fit(x_train, y_train)

model_svm_predict = model_svm.predict(x_test)
print(f"SVM model prediction is {model_svm_predict}")

acc_score_svm = accuracy_score(y_test, model_svm_predict)
print(f"The accuracy score of SVM model is {acc_score_svm}")
print(50*"_")



# Save My Best Model For Future Prediction -----------


joblib.dump(model_dt, "dt_risk_model.pkl")
joblib.dump(model_rf, "rf_risk_model.pkl")
joblib.dump(model_knn,"knn_model.pkl")
















