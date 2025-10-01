#Project 1 AER 850

#Part 1: Data Processing

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