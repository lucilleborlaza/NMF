# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:05:05 2023
Testing NMF for source apportionment. 
@author: lb945465
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#Load the data
df = pd.read_excel(r"C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\NMF\Queens_sorted_2000-2021.xlsx")

df.sort_values(by='date_local')
df.date_local.duplicated().sum()
df["date_local"].isnull().sum()

df.rename(columns={"date_local": "Date"}, inplace=True)
df.set_index('Date', inplace=True)

# Converting the index as date
df.index = pd.to_datetime(df.index)
df.columns

# remove special character
df.columns = df.columns.str.replace(',', '')

#Choose 2002-01-08 onwards (ONLY FOR QUEENS, ELIZABETH)
df=df[df.index > '2002-01-08'] #Queens

df.fillna(df.mean(), inplace=True)

#Create a dataframe of available samples
VOC_sample=df.drop(list(df.filter(regex='_dl')), axis=1)

# Create an NMF instance: 2 components
model = NMF(n_components=10, init='random', random_state=0, max_iter=1000)

# Apply NMF to the data matrix
W = model.fit_transform(VOC_sample)
H = model.components_

# Reconstruct the original matrix (approximation)
reconstructed_VOC = np.dot(W, H)

# Convert the matrices to DataFrames
W_df = pd.DataFrame(W, index=VOC_sample.index)
H_df = pd.DataFrame(H)
reconstructed_VOC_df = pd.DataFrame(reconstructed_VOC,  index=VOC_sample.index)

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter('NMF_results.xlsx', engine='xlsxwriter')

# Write each DataFrame to a different worksheet
W_df.to_excel(writer, sheet_name='Basis Matrix W')
H_df.to_excel(writer, sheet_name='Coefficient Matrix H')
reconstructed_VOC_df.to_excel(writer, sheet_name='Reconstructed Matrix')

# Close the Pandas Excel writer and output the Excel file
writer.save()

# Plot and save each component from the basis matrix W
for i, component in enumerate(W.T):
    plt.figure(figsize=(15, 5), dpi=150)
    plt.plot(VOC_sample.index, component)
    plt.title(f'Component {i+1} of Basis Matrix W')
    plt.xlabel('Date')
    plt.ylabel('Component Magnitude')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'Basis_Matrix_W_Component_{i+1}.png')  # Save figure
    plt.show()

# Plot and save each component from the coefficient matrix H
for i, component in enumerate(H):
    plt.figure(figsize=(15, 5), dpi=150)
    plt.bar(range(len(component)), component, tick_label=VOC_sample.columns)
    plt.title(f'Component {i+1} of Coefficient Matrix H')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Magnitude')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'Coefficient_Matrix_H_Component_{i+1}.png')  # Save figure
    plt.show()

# Sum the values of each column in W and create a pie chart
column_sums = W.sum(axis=0)

plt.figure(figsize=(8, 8), dpi=150)
plt.pie(column_sums, labels=[f'Component {i+1}' for i in range(len(column_sums))],
        autopct='%1.1f%%', startangle=140)
plt.title('Percentage Contribution of Each Component in Basis Matrix W')
plt.savefig('Basis_Matrix_W_Pie_Chart.png')  # Save figure
plt.show()

# # Load the dataframe with columns ws_ms and wd
# WS = pd.read_csv(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\NYSERDA VOC project\Data\Wind data\Requested_metdata_for_LA.GUARDIA.AIRPORT.csv', usecols=["date_LT", "ws_ms", "wd"])  
# WS.set_index("date_LT", inplace=True)
# WS.rename_axis("Date", inplace=True)
# WS.index = pd.to_datetime(WS.index)

# # Merge the NMF components with the wind data
# combined_df = pd.merge(W_df, WS, left_index=True, right_index=True, how='inner')
# combined_df.dropna(inplace=True)

# # Applying NPWR (Kernel Ridge Regression as an example)
# models = []
# for component in W_df.columns:
#     # Define the model
#     kr = KernelRidge(kernel='rbf')
#     # Fit the model
#     kr.fit(combined_df[['ws_ms', 'wd']], combined_df[component])
#     models.append(kr)

# for i, model in enumerate(models):
#     print(f"Model {i+1} Dual Coefficients:\n", model.dual_coef_)

# # Define parameter grid
# param_grid = {
#     'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': np.logspace(-3, 3, 7)  # Only used for rbf and poly kernels
# }

# # Grid search with cross-validation
# grid_search = GridSearchCV(KernelRidge(), param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(combined_df[['ws_ms', 'wd']], combined_df[W_df.columns[0]])

# # Best parameters
# print("Best Parameters:", grid_search.best_params_)

# # Retrain the model with the best parameters
# optimized_kr = KernelRidge(alpha=0.01, kernel='poly', gamma=100.0)
# optimized_kr.fit(combined_df[['ws_ms', 'wd']], combined_df[W_df.columns[0]])

# # Make predictions with the optimized model
# optimized_predictions = optimized_kr.predict(combined_df[['ws_ms', 'wd']])

# # Evaluate the optimized model
# mse_optimized = mean_squared_error(combined_df[W_df.columns[0]], optimized_predictions)
# r2_optimized = r2_score(combined_df[W_df.columns[0]], optimized_predictions)

# print(f"Optimized Model - MSE: {mse_optimized}, R-squared: {r2_optimized}")

