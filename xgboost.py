import os
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline

features_cache_path = input("Enter path to features_cache.npz file: ")
data = np.load(features_cache_path)
X_train, Y_train = data['X_train'], data['Y_train']
X_test, Y_test = data['X_test'], data['Y_test']

le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
Y_test_enc = le.transform(Y_test)
class_names = le.classes_

pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('xgb', xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=True,
        eval_metric='mlogloss'
    ))
])

pipeline.fit(X_train, Y_train_enc)
joblib.dump(pipeline, 'xgboost_emotion_model.joblib')
joblib.dump(le, 'label_encoder.joblib')

if os.path.exists('xgboost_emotion_model.joblib'):
    print(" xgboost_emotion_model.joblib has been saved.")
else:
    print("Model file was not created.")

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(Y_test_enc, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(Y_test_enc, y_pred, target_names=class_names))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(Y_test_enc, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix (Acc: {accuracy*100:.2f}%)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()