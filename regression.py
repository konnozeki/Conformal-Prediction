
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('ys1a.csv')
df = df[df.ys.notnull()]

features = ["vec", "delta", "deltachi", "deltahmix", "deltasmix"]
target = "ys"
seed = 123
x_train, x_remain, y_train, y_remain = train_test_split(df[features],
df[target], test_size=0.3,
random_state=7)
x_cal, x_test, y_cal, y_test = train_test_split(x_remain,
y_remain, test_size=0.50, random_state=7)

model = LinearRegression()
model.fit(x_train, y_train)

y_cal_pred = model.predict(x_cal)
y_cal_error = abs(y_cal - y_cal_pred)

quantile = y_cal_error.quantile(q=0.95, interpolation='midpoint')
print(quantile)

y_test_pred = model.predict(x_test)
lower = y_test_pred - quantile
upper = y_test_pred + quantile
index = y_test.index

coverage = [0]*len(y_test)
for i in range (len(index)):
    if (lower[i]<=y_test[index[i]]) and (upper[i]>=y_test[index[i]]):
        coverage[i] = 1
interval=pd.DataFrame(list(zip(y_test, lower,
y_test_pred, upper, coverage)),
columns=['actual', 'lower', 'predicted', 'upper', 'coverage'])
print('CoverRate:',sum(interval.coverage)/len(interval))