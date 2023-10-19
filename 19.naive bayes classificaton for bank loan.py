import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load the bank loan dataset (replace with your dataset)
loan_data = pd.read_csv('bank_loan_data.csv')

# Convert categorical features to numerical using LabelEncoder
label_encoder = LabelEncoder()
for column in loan_data.columns:
    if loan_data[column].dtype == 'object':
        loan_data[column] = label_encoder.fit_transform(loan_data[column])

# Select features and target variable
X = loan_data.drop('Loan_Status', axis=1)
y = loan_data['Loan_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes classifier
clf = GaussianNB()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=['No', 'Yes'])

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
