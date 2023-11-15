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
W_df = pd.DataFrame(W)
H_df = pd.DataFrame(H)
reconstructed_VOC_df = pd.DataFrame(reconstructed_VOC)

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter('NMF_results.xlsx', engine='xlsxwriter')

# Write each DataFrame to a different worksheet
W_df.to_excel(writer, sheet_name='Basis Matrix W')
H_df.to_excel(writer, sheet_name='Coefficient Matrix H')
reconstructed_VOC_df.to_excel(writer, sheet_name='Reconstructed Matrix')

# Close the Pandas Excel writer and output the Excel file
writer.save()

import matplotlib.pyplot as plt

# Plot and save each component from the basis matrix W
for i, component in enumerate(W.T):
    plt.figure(figsize=(10, 4), dpi=150)
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
    plt.figure(figsize=(10, 4), dpi=150)
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


