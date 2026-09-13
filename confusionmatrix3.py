from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np 

X,y = make_classification(
    n_samples=5000,
    n_features=10,
    weights= [0.97,0.03],
    random_state=42
)

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)
predicted = model.predict(X_test)

cm = confusion_matrix(y_test, predicted)
print("Confusion Matrix:")
print(cm)
print()

#Labelling every cell in the confusion matrix with its corresponding metric
tn,fp,fn,tp = cm.ravel()

print(f"True Negatives (TN): {tn} - correctly allowed legitimate transactions")
print(f"False Positives (FP): {fp} - Wrongly blocked legitimate transactions(angry customers)")
print(f"False Negatives (FN): {fn} - Missed Fraud (Fraudster got through)")
print(f"True Positives (TP): {tp} - Correctly caught fraud")
print()

#Bussiness Impact Translation

avg_transaction = 500 # Saudi Arabian Riyal(SAR)

print("=====  Bussiness Impact ========")
print(f"Revenue lost to false blocks : {fp * avg_transaction:,} SAR(Frustrated customers)")
print(f"Fraud losses Missed : {fn * avg_transaction:,} SAR(Fraud that got through)")
print(f"Fraud caught : {tp * avg_transaction:,} SAR(protected)")
print()

print(classification_report(y_test, predicted, target_names=['Legitimate','Fraud']))

