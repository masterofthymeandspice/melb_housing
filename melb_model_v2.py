import pandas as pd
import numpy as np
import random as random

melbourne_data = pd.read_csv('~/Documents/melb_housing/data/melb_data.csv')
print("Description of the melb data")
print(melbourne_data.describe())

print("print all columns in the data:")
print(melbourne_data.columns)


melbourne_data = melbourne_data.dropna(axis = 0)

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor


melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model
melbourne_model.fit(x,y)

from sklearn.metrics import mean_absolute_error

model_compare = pd.DataFrame({'actualPrice': melbourne_data['Price'], 'predictedPrice': melbourne_model.predict(x)})

predicted_home_prices = melbourne_model.predict(x)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(x,y, random_state=0)
melbourne_model = DecisionTreeRegressor()
#
# fit model

melbourne_model.fit(train_X, train_y)


# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 10)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


df_output = []

for max_leaf_nodes in range(2,1500):
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    df_output.append([max_leaf_nodes, my_mae])

cols = ['leafNodes', 'mae']
df_output = pd.DataFrame(df_output, columns=cols)



min_node = np.where(df_output["mae"] == df_output["mae"].min())

print(df_output.iloc[min_node])


import plotly.graph_objects as go
fig = go.Figure(data = go.Scatter(x=df_output.leafNodes, y = df_output.mae))
fig.update_xaxes(type="log") # log range: 10^0=1, 10^5=100000
fig.show()



















