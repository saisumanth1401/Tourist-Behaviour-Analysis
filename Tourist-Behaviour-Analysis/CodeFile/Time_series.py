import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
%matplotlib inline
import calendar
import sklearn
import sklearn.linear_model as lm

# Opening CSV File corresponding to a particular region
location = "Results/Input records/Filtered1M.csv"
df = pd.read_csv(location, header=None)
df2 = pd.to_datetime(df[5])
months = list(range(1, 13))
df[5] = pd.to_datetime(df[5])
ct = []

# Finding data wrt to each Month
for i in months:
    ct.append(len(df.loc[(df[5].dt.month == i)]))
for i in range(len(ct)):
    ct[i] = ct[i] / 10

months = np.asarray(months)
count = np.asarray(ct)
plt.scatter(months, count)
plt.plot(months, count, color='blue', linewidth=1)
months = list(range(1, 13))
year = list(range(2004, 2014))
allmonths = {}

for i in months:
    allmonths[i] = []
for i in year:
    for j in months:
        allmonths[j].append(len(df.loc[operator.and_(df[5].dt.year == i, df[5].dt.month == j)]))

# Checking value for April
regr = lm.LinearRegression()
sc = []

# Plotting data for each month to find Trend for that particular month

for month in months:
    monthCt = np.asarray(allmonths[month])
    yr = np.asarray(year)

    plt.scatter(year, monthCt)

    poly = make_pipeline(PolynomialFeatures(3), regr)

    poly.fit(yr.reshape(-1, 1), monthCt)

    Y_pred = poly.predict(yr.reshape(-1, 1).reshape(-1, 1))

    plt.plot(year, monthCt, color="orange", linewidth=1)
    plt.plot(year, Y_pred.reshape(-1, 1), color='blue', linewidth=1)

    month_diff = []
    for i in range(len(monthCt)):
        month_diff.append(abs(monthCt[i] - Y_pred[i]))

    sc.append(mean_absolute_error(monthCt, Y_pred))

    plt.title("Month: {}".format(month))
    mean_absolute_error(monthCt, Y_pred)

# Plot of Months with average no. of visitors
fig, ax = plt.subplots(figsize=(10, 5))
month_average_diff = [np.mean(diff) if isinstance(diff, list) else diff for diff in month_diff]

for month in range(1, 13):
    if month <= len(month_diff) and isinstance(month_diff[month-1], list):
        ax.scatter(month, np.asarray(month_diff[month-1]))
    elif month <= len(month_diff):
        ax.scatter(month, month_diff[month-1])
    
ax.scatter(months, np.asarray(sc))
ax.plot(months, np.asarray(sc), color='blue', linewidth=3)

ax.set_xlabel("Months")
ax.set_ylabel("Average count of tourists")

ax.plot(months, np.asarray(sc), color='blue', linewidth=3)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(months, np.asarray(sc))
ax.set_xlabel("Months")
ax.set_ylabel("Seasonal Component")
ax.plot(months, np.asarray(sc), color='blue', linewidth=1)
