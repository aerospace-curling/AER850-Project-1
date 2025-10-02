#Project 1 AER 850

#Part 1: Data Processing and splitting the data into test and train data

#reading the data and importing the pandas library
import pandas as pd
data=pd.read_csv("data/Project 1 Data.csv")
#To check that reading the data is working correctly:
#print(data.head())

# Remove any data that is missing in the set
data = data.dropna().reset_index(drop=True)

#The data must be split into test and train data, which will be completed with the use of Stratified Shuffle Split
#There is no requirement to put the data into bins being that it is already sorted into "Steps"
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["Step"]):
    strat_data_train = data.loc[train_index].reset_index(drop=True)
    strat_data_test = data.loc[test_index].reset_index(drop=True)


#to check if the splitting worked as was expected, the "Step" proportions are checked
print("\nBelow is a check for the proportions of 'Steps' in the test data set:")
print(strat_data_test["Step"].value_counts() / len(strat_data_test))
print("\nThis is compared to the full data set:")
print(data["Step"].value_counts() / len(data))

#with the test and full data set having similar proprotions, it can be confirmed that the splitting was generally effective

#We need to separate the target "Step" from the features
y_train = strat_data_train['Step']
x_train = strat_data_train.drop(columns=['Step'])
y_test = strat_data_test['Step']
x_test = strat_data_test.drop(columns=['Step'])

#Part 2: Data Visualization

#importing the modules
import numpy as np
import matplotlib.pyplot as plt

#Obtaining statistical data about the dataset and printing it
print("This is statistical data about the dataset:\n")
print(strat_data_train.describe())

#Creating an array of the items to be compared with the subplots
items = ['X','Y','Z','Step']

#creating a plot with subplots to show the trends in data. The size of the plot is defined by the length of the matrix, Items
fig, axs = plt.subplots(len(items),len(items))


  
#creating the subplots through the use of loops
for i in range(len(items)):
    for j in range(len(items)):
        if i ==j:
            #creating the histograms with 20 bins each
            axs[i,j].hist(strat_data_train[items[i]], bins=20)
        else:
                #creating the scatter plots
                axs[i,j].scatter(strat_data_train[items[j]], strat_data_train[items[i]], s=4)
        if j==0:
                #this part is used to label the axes
                axs[i,j].set_ylabel(items[i])
        else:
                    #removes the y-ticks except those on the first column
                    axs[i,j].set_yticks([])
        if i == len(items) - 1:
                        #if the subplot is in the last row, the x-axis label is the name at position j
                        axs[i, j].set_xlabel(items[j])
        else:
                            #removes x-axis ticks except those on the last row
                            axs[i, j].set_xticks([])
                            


#Step 3: Correlation Analysis

#importing the seaborn library
import seaborn as sns

#Computing the Pearson Correlation matrix
corr_matrix = strat_data_train.corr(method='pearson')

#Printing the correlation 
print("Correlation of features related to the Step:\n",corr_matrix["Step"])

#Making the heatmap to make the correlation more identifiable
plt.figure()
sns.heatmap(np.abs(corr_matrix))
plt.title("Heatmap of Correlation Matrix")
plt.show()

#in order to determine if there are any highly collinear components, a masked correlation matrix is created
# the threshold determined is 0.80
plt.figure()
plt.title('Masked Heatmap with Threshold of 0.80')
corr_matrix_mask = np.abs(corr_matrix) < 0.80
sns.heatmap(corr_matrix_mask)
plt.show()

#Step 4 Classification Model Development/Engineering

#The first ML model used will be LogisticRegression through a Pipeline with StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

cv= StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

#creating the pipeline
pipeline1 = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=5000,random_state=42))])
param_grid = {'model__C': [0.1, 1, 10, 100, 1000, 10000, 100000],'model__penalty': ['l2']}
#Using GridSearch
grid_search=GridSearchCV(pipeline1,param_grid,cv=cv, n_jobs=-1, refit=True, verbose=1)

grid_search.fit(x_train, y_train)
 
# The best parameters and score are printed out
print("\nThe best parameters are:", grid_search.best_params_)
print("\nThe best score is:", grid_search.best_score_)



                  