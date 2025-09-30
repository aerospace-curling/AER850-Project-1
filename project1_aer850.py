#Project 1 AER 850

#Part 1: Data Processing

#reading the data and importing the pao"ndas library
import pandas as pd
data=pd.read_csv("data/Project 1 Data.csv")
#To check that reading the data is working correctly:
#print(data.head())


#Part 2: Data Visualization

#importing the modules
import numpy as np
import matplotlib.pyplot as plt

#Creating a line graph of the Step number vs. movement in the Z direction
plt.figure()
plt.plot(data["Step"], data["Z"], color="blue", marker="o")
plt.title("Z-axis Progression Across Steps")
plt.xlabel("Step Number")
plt.ylabel("Z-axis Progression")
plt.show

#Creating a line graph of the Step number vs. movement in the Y direction
plt.figure()
plt.plot(data["Step"], data["Y"], color="black", marker="o")
plt.title("Y-axis Progression Across Steps")
plt.xlabel("Step Number")
plt.ylabel("Y-axis Progression")
plt.show

#Creating a line graph of the Step number vs. movement in the X direction
plt.figure()
plt.plot(data["Step"], data["X"], color="green", marker="o")
plt.title("X-axis Progression Across Steps")
plt.xlabel("Step Number")
plt.ylabel("X-axis Progression")
plt.show

#Creating a scatter plot
plt.figure()
plt.scatter(data["Step"], data["Z"], c=data["Step"], cmap="Blues")
plt.title("Scatterplot of Z-movement across the Steps")
plt.xlabel("Step Number")
plt.ylabel("Z-axis movement")
plt.show()
