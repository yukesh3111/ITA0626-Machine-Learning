import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Generate some example data
# You should replace this with your actual dataset
np.random.seed(42)
X = np.random.rand(100, 2)  # Features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naïve Bayes classifier
naive_bayes = GaussianNB()

# Train the classifier on the training data
naive_bayes.fit(X_train, y_train)

# Make predictions on the test data
y_pred = naive_bayes.predict(X_test)

# Calculate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and display the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
