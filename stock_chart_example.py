import json
import pandas as pd
import matplotlib.pyplot as plt

# I have loaded the stock Information file onto to dataExtract
with open('stock_information.json', 'r') as file:
    dataExtract = json.load(file)

# Extracted the time series from the loaded file 
time_series = dataExtract['Time Series (Daily)']

# Convert the time series data to a DataFrame
dF = pd.DataFrame.from_dict(time_series, orient='index')

# Convert the index to datetime
dF.index = pd.to_datetime(dF.index)

# Sort the DataFrame in an ascending order
dF = dF.sort_index()

# Conversion to numeric values to plot
dF = dF.apply(pd.to_numeric)

# Using matplotlib to plot the open price vs volume

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot for Open Price vs Volume with transparency
ax1.scatter(dF['1. open'], dF['5. volume'], color='red', alpha=0.5, label='Open Price vs Volume')
ax1.set_xlabel('Open Price')
ax1.set_ylabel('Volume')
ax1.set_title('Volume vs Open Price')


# Scatter plot for Close Price vs Volume with transparency
ax2.scatter(dF['4. close'], dF['5. volume'], color='black', alpha=0.5, label='Close Price vs Volume')
ax2.set_xlabel('Close Price')
ax2.set_ylabel('Volume')
ax2.set_title('Volume vs Close Price')
plt.show()
