import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Load the data from the Excel files
train_file_path = '/content/drive/MyDrive/ResoluteAI/Aastha Sharma Task 1/train.xlsx'
test_file_path = '/content/drive/MyDrive/ResoluteAI/Aastha Sharma Task 1/test.xlsx

# Read the train and test datasets
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)