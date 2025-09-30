#Project 1 AER 850

#Part 1: Data Processing

#reading the data and importing the pandas library
import pandas as pd
data=pd.read_csv("data/Project 1 Data.csv")
#To check that reading the data is working correctly:
#print(data.head())


#Part 2: Data Visualization

#importing the modules
import numpy as np
import matplotlib.pyplot as plt

#Obtaining statistical data about the dataset and printing it
print("This is statistical data about the dataset:\n")
print(data.describe())

#Creating a scatter plot for movement in the z-axis versus the step number
plt.figure()
plt.scatter(data["Step"], data["Z"], c=data["Step"])
plt.title("Scatterplot of Z-movement across the Steps")
plt.xlabel("Step Number")
plt.ylabel("Z-axis movement")
plt.show()

#Creating a scatter plot for movement in the Y-axis versus the step number
plt.figure()
plt.scatter(data["Step"], data["Y"], c=data["Step"])
plt.title("Scatterplot of Y-movement across the Steps")
plt.xlabel("Step Number")
plt.ylabel("Y-axis movement")
plt.show()

#Creating a scatter plot for movement in the X-axis versus the step number
plt.figure()
plt.scatter(data["Step"], data["X"], c=data["Step"])
plt.title("Scatterplot of X-movement across the Steps")
plt.xlabel("Step Number")
plt.ylabel("X-axis movement")
plt.show()

#creating a histogram for the Z-movement to show distribution of movement size
plt.figure()
plt.hist(data['Z'], bins=20)
plt.title("Histogram of Z-axis Movement Values")
plt.xlabel("Z-axis movement")
plt.ylabel("Frequency")
plt.show

#creating a histogram for the X-movement to show distribution of movement size
plt.figure()
plt.hist(data['X'], bins=20)
plt.title("Histogram of X-axis Movement Values")
plt.xlabel("X-axis movement")
plt.ylabel("Frequency")
plt.show

#creating a histogram for the Y-movement to show distribution of movement size
plt.figure()
plt.hist(data['Y'], bins=20)
plt.title("Histogram of Y-axis Movement Values")
plt.xlabel("Y-axis movement")
plt.ylabel("Frequency")
plt.show


#Step 3: Correlation Analysis

#importing the seaborn library
import seaborn as sns

#Computing the Pearson Correlation matrix
corr_matrix = data.corr(method='pearson')

#Printing the correlation 
print("Correlation of features related to the Step:\n",corr_matrix["Step"])

#Making the heatmap to make the correlation more identifiable
plt.figure()
sns.heatmap(np.abs(corr_matrix))
plt.title("Heatmap of Correlation Matrix")
plt.show