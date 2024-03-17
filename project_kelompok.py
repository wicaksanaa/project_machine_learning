# %% [markdown]
# ## **Project Kelompok** 
# ##### **Relasi antara Salary dan Years Experience**

# %% [markdown]
# ##### 1. Natasya Salsabila
# ##### 2. Agnes Situmorang
# ##### 3. Islam Cahya Wicaksana

# %% [markdown]
# # Import Library

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# # Data Gathering

# %%
data = pd.read_csv('Salary_Data.csv')

# %% [markdown]
# # EDA

# %%
data.info()

# %%
data.head()

# %% [markdown]
# # Main Code

# %% [markdown]
# #### Tampung Column pada Variable

# %%
# Separate attribute and label
X = data['YearsExperience']
y = data['Salary']

# %%
# Change the attribute form
X = np.array(X)
X = X[:,np.newaxis]

# %% [markdown]
# #### Build Model SVM

# %%
# Build a model with C, gamma, and kernel parameters
model = SVR()
parameters = {
    'kernel': ['rbf'],
    'C': [1000, 10000, 100000],
    'gamma': [0.5, 0.05, 0.005]
}
grid_search = GridSearchCV(model, parameters)

# %%
# Train the model with the fit function
grid_search.fit(X,y)

# %%
# Displays the best parameters of the grid_search object
print(grid_search.best_params_)

# %%
# Create a new SVM model with the best parameters from the grid search results
svm_model = SVR(C=1000000, gamma=0.005, kernel='rbf')
svm_model.fit(X,y)

# %% [markdown]
# #### Build Model Regresi Linear

# %%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
regresi = LinearRegression()
regresi.fit(x_train, y_train)

# %%
y_pred = regresi.predict(x_test)

# %% [markdown]
# #### Build Model Random Forest

# %%
model_forest = RandomForestRegressor()

# %%
model_forest.fit(x_train, y_train)

# %%
score = model_forest.score(x_test, y_test)
print(f"Random Forest Model Score: {score}")

# %% [markdown]
# #### Plotting & Comparing

# %%
score_linear_regression = regresi.score(x_test, y_test)
score_random_forest = model_forest.score(x_test, y_test)
score_svm = svm_model.score(x_test, y_test)

print("Linear Regression Score:", score_linear_regression)
print("Random Forest Score:", score_random_forest)
print("SVM Score:", score_svm)

# %%
# Plot data as scatter plot
plt.scatter(X, y, color='black', label='Data')

# Plot linear regression predictions
plt.plot(X, regresi.predict(X), color='blue', label='Linear Regression')

# Plot random forest predictions
plt.plot(X, model_forest.predict(X), color='green', label='Random Forest')

# Plot SVM predictions
plt.plot(X, svm_model.predict(X), color='red', label='SVM')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Comparison of Models')
plt.legend()
plt.show()


