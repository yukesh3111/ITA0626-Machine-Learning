import pandas as pd

# Sample training data
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['High', 'High', 'High', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Initialize specific and general hypotheses
specific_hypothesis = df.iloc[0, :-1].tolist()
general_hypothesis = [['?' for _ in range(len(df.columns) - 1)]]

# Iterate through the training examples
for index, row in df.iterrows():
    if row['EnjoySport'] == 'Yes':
        for i in range(len(specific_hypothesis)):
            if specific_hypothesis[i] != row[i]:
                specific_hypothesis[i] = '?'
                for g in general_hypothesis:
                    if g[i] == row[i]:
                        general_hypothesis.remove(g)
            else:
                for g in general_hypothesis:
                    if g[i] != row[i] and g[i] != '?':
                        general_hypothesis.remove(g)
    else:
        new_general_hypotheses = []
        for i in range(len(specific_hypothesis)):
            if specific_hypothesis[i] != row[i] and specific_hypothesis[i] != '?':
                new_general_hypothesis = specific_hypothesis.copy()
                new_general_hypothesis[i] = '?'
                new_general_hypotheses.append(new_general_hypothesis)
        general_hypothesis.extend(new_general_hypotheses)
        general_hypothesis = [list(x) for x in set(tuple(x) for x in general_hypothesis)]

print("Specific Hypothesis:", specific_hypothesis)
print("General Hypotheses:", general_hypothesis)
