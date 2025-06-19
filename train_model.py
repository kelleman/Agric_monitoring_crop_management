import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

# Load data
df = pd.read_csv('data/crop_data.csv')

# Split features and target
X = df.drop('Crop', axis=1)
y = df['Crop']

# Encode the Crop labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")

# Save model and label encoder
joblib.dump(model, 'crop_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Plot and save accuracy score as a bar
plt.figure(figsize=(4, 4))
plt.bar(['Accuracy'], [acc], color='green')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.savefig('accuracy.png')
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("✅ Model, label encoder, and performance visuals saved.")
