import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('huang-401.csv')
le = LabelEncoder()
df['phase'] = le.fit_transform(df['phase'].to_numpy())
features = ["vec", "delta", "deltachi", "deltahmix", "deltasmix"]
target = "phase"
seed = 123
x_train, x_remain, y_train, y_remain= train_test_split(df[features], df[target], test_size=0.25, random_state=seed)
x_cal, x_test, y_cal, y_test = train_test_split(x_remain, y_remain, test_size=0.50, random_state=seed)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

n = len(x_cal)
preds = model.predict_proba(x_cal)
prob_true = preds[np.arange(n), y_cal]
scores = 1 - prob_true


alpha = 0.05
q_level = np.ceil((n + 1) * (1 - alpha)) / n
qhat = np.quantile(scores, q_level, method='higher')
prediction_sets = (1 - model.predict_proba(x_test) <= qhat)
real = le.classes_[df.phase[y_test.index]]
predset = [''] * len(y_test)
coverage = [0] * len(y_test)
for i in range(len(y_test)):
    predset[i] = ','.join(le.classes_[prediction_sets[i]])
    if real[i] in predset[i]:
        coverage[i] = 1

res = pd.DataFrame(list(zip(real, predset, coverage)), columns=['actual', 'predicted', 'coverage'])
print("Cover rate =", sum(res.coverage) / len(res))


