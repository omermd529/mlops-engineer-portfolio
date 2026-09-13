import joblib
import json
import os
import hashlib
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,recall_score, precision_score

def train_and_save_model(model,model_name,X_train,X_test,y_train,y_test):
    """Train a model,evaluate it,save it with metatdata.
    This is what your kubeflow pipeline training component does."""
    
    #Train
    model.fit(X_train,y_train)
    
    #Evaluate
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'auc_roc': round(roc_auc_score(y_test,y_proba), 4),
        'recall': round(recall_score(y_test,y_pred), 4),
        'precision': round(precision_score(y_test,y_pred, zero_division=0), 4),
    }
    
    #Save Model
    
    os.makedirs('models', exist_ok = True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/{model_name}_{timestamp}.pk1'
    joblib.dump(model, model_path)
    
    #Save metatdata alongsidemodel- critical for MLflow style tracking
    
    metadata = {
        'model_name': model_name,
        'model_path': model_path,
        'trained_at':timestamp,
        'metrics':metrics,
        'model_type': type(model).__name__,
        'file_size_kb':round(os.path.getsize(model_path)/1024,2),
        'checksum': hashlib.md5(open(model_path,'rb').read()).hexdigest()
        
    }
    
    meta_path = model_path.replace('.pk1','metadata.json')
    with open(meta_path,'w') as f:
        json.dump(metadata,f,indent=2)
        
    print(f"Saved: {model_path}")
    print(f"Metrics: {metrics}")
    print(f"Size: {metadata['file_size_kb']} KB")
    print(f"Checksum: {metadata['checksum'][:8]}")
    print()
    
    return model_path,metrics

def load_and_serve(model_path,transaction):
    """Load a saved model and serve a single prediction.
    This is what your KServe inference container does at startup + per request"""
    
    # Startup Load Model once 
    
    model= joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    
    #per request :predict
    
    import numpy as np 
    X = np.array(transaction).reshape(1, -1)
    fraud_proba = model.predict_proba(X)[0][1]
    
    response = {
        'fraud_probability': round(float(fraud_proba),4),
        'decision': 'BLOCK'if fraud_proba > 0.5 else 'ALLOW',
        'risk_level': 'HIGH'if fraud_proba > 0.7 else 'MEDIUM' if fraud_proba > 0.3 else 'LOW',
        'model_version': model_path.split('/')[-1]
        
    }
    
    return response

# Generate Data

X,y = make_classification(n_samples=5000, n_features=10,weights=[0.97,0.03],random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Train two versions - Champion and Challenger

print("====== Training Champion(Random Forest) ======")
rf_path,rf_metrics = train_and_save_model((RandomForestClassifier(n_estimators=100, random_state=42)),
'fraud_rf', X_train, X_test, y_train, y_test)

print("====== Training Challenger(Gradient Boosting) ======")
gb_path,gb_metrics = train_and_save_model(GradientBoostingClassifier(n_estimators=100,random_state=42),
                                          'fraud_gb', X_train, X_test, y_train, y_test)

# Serve a prediction from the champion 
print("====== Serving a prediction =========")
sample_transaction = X_test[0].tolist()
response = load_and_serve(rf_path, sample_transaction)
print(json.dumps(response,indent=2))
