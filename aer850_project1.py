#Project 1 AER 850
#Emily Peelar 501169755

#Project 1 AER 850
#Emily Peelar 501169755

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
from sklearn.model_selection import StratifiedShuffleSplit
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
corr_matrix_mask = np.abs(corr_matrix) >= 0.80
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
param_grid = {'model__C': np.logspace(-3, 1, 3),'model__penalty': ['l2']}
#Using GridSearch
#the best score using the line below is 0.9835053959522382
#grid_search=GridSearchCV(pipeline1,param_grid,cv=cv, n_jobs=-1, refit=True, verbose=1, scoring='f1_weighted')
#the best score from below is 0.9840262350576536, which is slightly improved
grid_search=GridSearchCV(pipeline1,param_grid,cv=cv, n_jobs=-1, refit=True, verbose=1, scoring='f1_weighted')

grid_search.fit(x_train, y_train)
 
# The best parameters and score are printed out
print("\nThe best parameters are:", grid_search.best_params_)
print("\nThe best score is:", grid_search.best_score_)




#Now trying the second ML model, SVM
from sklearn.svm import SVC

#creating the pipeline for SVM
pipeline_SVM = Pipeline([('scaler', StandardScaler()),('model', SVC(random_state=42))])

#this one is not as good, score is 0.9912937691738073    
#SVM_param_grid={'model__kernel': ['linear'],'model__C':np.logspace(-3, 3, 5),'model__gamma': np.logspace(-3, 1, 3)}
#this one is not as good, score is 0.9840262350576536
#SVM_param_grid={'model__kernel': ['linear'],'model__C':np.logspace(-2, 3, 6),'model__gamma': np.logspace(-2, 3, 6)}
#this one is the best , score is 0.9926632848562396
SVM_param_grid={'model__kernel': ['linear','rbf'],'model__C':np.logspace(-3, 3, 5),'model__gamma': np.logspace(-3, 1, 3)}

SVM_grid_search=GridSearchCV(pipeline_SVM,SVM_param_grid,cv=cv, n_jobs=-1, refit=True, verbose=1,scoring='f1_weighted')

SVM_grid_search.fit(x_train, y_train)    

# The best parameters and score are printed out
print("\nThe best parameters for SVM are:", SVM_grid_search.best_params_)
print("\nThe best score for SVM is:", SVM_grid_search.best_score_) 



 

#Now trying RandomForest model

from sklearn.ensemble import RandomForestClassifier

#creating the pipeline for RandomForest
#there is no need for Standardscaler because the tree splits based on thresholds
pipeline_randomforest = Pipeline([('model', RandomForestClassifier(random_state=42))])      

#the score from below is 0.992743044536126
#randomforest_param_grid = { 'model__n_estimators': [100],'model__criterion': ['gini'], 'model__max_depth': [None],'model__min_samples_split': [2, 5, 10],'model__min_samples_leaf': [1, 2, 4],'model__max_features': ['sqrt', 'log2', None] }
#the best is obtained with the code below which is 0.9955537084892846
randomforest_param_grid = { 'model__n_estimators': [10,100,150],'model__criterion': ['gini','entropy'], 'model__max_depth': [2,5,10],'model__min_samples_split': [2, 5, 10],'model__min_samples_leaf': [2,5,10],'model__max_features': ['sqrt', 'log2', None] }

randomforest_grid_search=GridSearchCV(pipeline_randomforest,randomforest_param_grid,cv=cv, n_jobs=-1, refit=True, verbose=1, scoring='f1_weighted')

randomforest_grid_search.fit(x_train, y_train)   

# The best parameters and score are printed out
print("\nThe best parameters for RandomForest are:", randomforest_grid_search.best_params_)
print("\nThe best score for RandomForest is:", randomforest_grid_search.best_score_) 




#Now trying DecisionTree with RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

#creating the pipeline for DecisionTree
pipeline_decisiontree= Pipeline([('model', DecisionTreeClassifier(random_state=42))])



#score from below .9814727773294918
#decisiontree_param_distributions={'model__criterion': ['gini'], 'model__max_depth': [None],'model__min_samples_split': [2, 5, 10],'model__min_samples_leaf': [1, 2, 4],'model__max_features': ['sqrt', 'log2'] }
#score from below is the better score, which will be used 0.9831771512574269
decisiontree_param_distributions={'model__criterion': ['gini','entropy'], 'model__max_depth': [5,10,20],'model__min_samples_split': [2,5,10],'model__min_samples_leaf': [1, 2, 3],'model__max_features': ['sqrt', 'log2'] }

decisiontree_random_search = RandomizedSearchCV(estimator=pipeline_decisiontree,param_distributions=decisiontree_param_distributions,cv=cv,n_jobs=-1,random_state=42,verbose=1, scoring='f1_weighted')

decisiontree_random_search.fit(x_train, y_train)

# The best parameters and score are printed out
print("\nThe best parameters for Decision Tree with RandomSearch are:", decisiontree_random_search.best_params_)
print("\nThe best score for Decision Tree with Random Search is:", decisiontree_random_search.best_score_) 






#Starting Step 5: Model Performance Analysis
from sklearn.metrics import f1_score


#starting with the f1score

#starting with logistic regression
bestmodel_logisticregression=grid_search.best_estimator_
y_pred = bestmodel_logisticregression.predict(x_test)

f1_weighted_logisticregression = f1_score(y_test, y_pred, average='weighted')
print("\nThe f1 score of logistic regression is:", f1_weighted_logisticregression)

#now with SVM
bestmodel_SVM=SVM_grid_search.best_estimator_
y_pred_SVM = bestmodel_SVM.predict(x_test)

f1_weighted_SVM = f1_score(y_test, y_pred_SVM, average='weighted')
print("\nThe f1 score of SVM is:", f1_weighted_SVM)

#now with the RandomForest model
bestmodel_randomforest=randomforest_grid_search.best_estimator_
y_pred_randomforest = bestmodel_randomforest.predict(x_test)

f1_weighted_randomforest = f1_score(y_test, y_pred_randomforest, average='weighted')
print("\nThe f1 score of the RandomForest is:", f1_weighted_randomforest)

#now with the DecisionTree Model
bestmodel_decisiontree=decisiontree_random_search.best_estimator_
y_pred_decisiontree = bestmodel_decisiontree.predict(x_test)

f1_weighted_decisiontree = f1_score(y_test, y_pred_decisiontree, average='weighted')
print("\nThe f1 score of the DecisionTree is:", f1_weighted_decisiontree)




#now doing the precision score
from sklearn.metrics import precision_score

precision_weighted_logistic = precision_score(y_test, y_pred, average='weighted')
print("\nThe precision score of the Logistic Regression is:", precision_weighted_logistic)

precision_weighted_SVM = precision_score(y_test, y_pred_SVM, average='weighted')
print("\nThe precision score of the SVM is:", precision_weighted_SVM)

precision_weighted_randomforest = precision_score(y_test, y_pred_randomforest, average='weighted')
print("\nThe precision score of the RandomForest is:", precision_weighted_randomforest)

precision_weighted_decisiontree = precision_score(y_test, y_pred_decisiontree, average='weighted')
print("\nThe precision score of the DecisionTree is:", precision_weighted_decisiontree)



#now doing the accuracy score
from sklearn.metrics import accuracy_score

accuracy_logistic = accuracy_score(y_test, y_pred)
print("\nThe accuracy score of the Logistic Regression is:", accuracy_logistic)

accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
print("\nThe accuracy score of the SVM is:", accuracy_SVM)

accuracy_randomforest = accuracy_score(y_test, y_pred_randomforest)
print("\nThe accuracy score of the RandomForest is:", accuracy_randomforest)

accuracy_decisiontree = accuracy_score(y_test, y_pred_decisiontree)
print("\nThe accuracy score of the DecisionTree is:", accuracy_decisiontree)




#now creating the confusion matrix with random forest as the best machine learning method

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

confusionmatrix_randomforest=confusion_matrix(y_test,y_pred_randomforest)
print("\nRandom Forest Confusion Matrix:\n", confusionmatrix_randomforest)

disp = ConfusionMatrixDisplay(confusion_matrix=confusionmatrix_randomforest,display_labels=range(1, 14))
disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()





#Step 6: Stacked Model Performance Analysis

#starting with using the stacked model and importing the StackingClassifier
#logistic regression and random forest are used because they both have accuracy, f1 scores, and precision of 1. They are also very different types of classifiers, which is important for the stacked model
from sklearn.ensemble import StackingClassifier

estimators = [("rf", randomforest_grid_search.best_estimator_),("lr", grid_search.best_estimator_)   ]


classifier = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(max_iter=5000,random_state=42), n_jobs=-1)

classifier.fit(x_train,y_train)

prediction_y_stacked= classifier.predict(x_test)

#determining the scores
f1_weighted_stacked = f1_score(y_test, prediction_y_stacked, average='weighted')
print("\nThe f1 score of the Stacked Model is:", f1_weighted_stacked)

precision_weighted_stacked = precision_score(y_test, prediction_y_stacked, average='weighted')
print("\nThe precision score of the Stacked Model is:", precision_weighted_stacked)

accuracy_stacked = accuracy_score(y_test, prediction_y_stacked)
print("\nThe accuracy score of the Stacked Model is:", accuracy_stacked)

#now to make the confusion matrix
confusionmatrix_stacked = confusion_matrix(y_test,prediction_y_stacked)
print("\n Stacked Model Confusion Matrix:\n", confusionmatrix_stacked)

disp=ConfusionMatrixDisplay(confusion_matrix=confusionmatrix_stacked,display_labels=range(1,14))
disp.plot(cmap="Blues")
plt.title("Stacked Model Confusion Matrix")
plt.show()



#Step 7:
    
import joblib
finalized_model = randomforest_grid_search.best_estimator_

joblib.dump(finalized_model, "finalized_model_AER815_project1.joblib")

model = joblib.load("finalized_model_AER815_project1.joblib")

coordinates = np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8],[9.4,3,1.3]])

coordinates_dataframe = pd.DataFrame(coordinates, columns=["X","Y","Z"])

predicted_steps = model.predict(coordinates)

print("\n The predicted values for the matrices are summarized below:\n")
for row in coordinates:
    predicted_step = model.predict([row])[0]   # predict needs 2D input
    print(row, "is predicted to be Step", predicted_step)