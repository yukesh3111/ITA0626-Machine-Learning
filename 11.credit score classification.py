import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

# Generate sample data (replace this with your actual data)
np.random.seed(42)
num_samples = 1000
num_features = 5
X = np.random.rand(num_samples, num_features)
y = np.random.randint(2, size=num_samples)  # Binary labels: 0 or 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with standardization and SVM classifier
clf = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42))
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
