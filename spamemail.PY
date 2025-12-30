import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
data = {
    'message': [
        'you won lottery',
        'extra income',
        'win cash prize',
        'limited offer',
        'claim your free reward',
        'urgent cash bonus',
        'click link to win',
        'exclusive deal for you',
        'meeting at 5 pm',
        'please call me',
        'are you coming today',
        'project deadline tomorrow',
        'let us study together',
        'send me the notes',
        'happy birthday'
    ],
    'label': [
        'spam',
        'spam',
        'spam',
        'spam',
        'spam',
        'spam',
        'spam',
        'spam',
        'not spam',
        'not spam',
        'not spam',
        'not spam',
        'not spam',
        'not spam',
        'not spam'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode labels
df['label'] = df['label'].map({'spam': 1, 'not spam': 0})

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
Evaluate=model.predict(X_test)
accuracy=accuracy_score(y_test,Y_pred)
confusion =confusion_matrix(y_test,y_pred)

#output
#input we give 
new_email=["limited time offer"]
new_email_vectorized = vectorizer.transform(new_email)
prediction = model.predict(new_email_vectorized)
if prediction[0] == 1:
    print("The email is spam.")
else:
    print("The email is not spam.")