import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# load the dataset
df = sns.load_dataset('tips')

x = df[['total_bill', 'size']]
y = df['tip']

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)


# train the model
model = LinearRegression()
model.fit(x_train, y_train)

# save the model
joblib.dump(model, 'model.joblib')
print('Model trained and saved successfully')