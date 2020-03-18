# Assignment_1 Intelligent Data Analysis 

# Importing the libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the data set-2
dataset2 = pd.read_csv('Biomechanical_Data_2Classes.csv')
X2 = dataset2.iloc[:, :-1].values
y2 = dataset2.iloc[:, 6:7].values

#Encoding Categorical Value 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y2[:, 0] = labelencoder.fit_transform(y2[:, 0])

# Spliiting the data set as training and test sets 
from sklearn.model_selection import train_test_split 
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 8/31, random_state = 0)
y2_train = y2_train.astype('int')
y2_test = y2_test.astype('int')

# Importing classes for decison tree an confusiin matrix 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# Creating decision trees “minimum records per leaf node” values of 5
classifier5 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 5, random_state = 0)
## Fitting the Data Set leaf node = 5 
classifier5 = classifier5.fit(X2_train, y2_train)
y5_pred = classifier5.predict(X2_test)
##Creating confusion Matrix leaf node = 5 
cm5 = confusion_matrix(y2_test, y5_pred)


# Creating decision trees “minimum records per leaf node” values of 15
classifier15 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 15, random_state = 0)
## Fitting the Data Set leaf node = 15 
classifier15 = classifier15.fit(X2_train, y2_train)
y15_pred = classifier15.predict(X2_test)
## Creating confusion Matrix leaf node = 15 
cm15 = confusion_matrix(y2_test, y15_pred)


# Creating decision trees “minimum records per leaf node” values of  25
classifier25 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 25, random_state = 0)
## Fitting the Data Set leaf node = 25 
classifier25 = classifier25.fit(X2_train, y2_train)
y25_pred = classifier25.predict(X2_test)
## Creating confusion Matrix leaf node = 25 
cm25 = confusion_matrix(y2_test, y25_pred)


# Creating decision trees “minimum records per leaf node” values of  40
classifier40 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 40, random_state = 0)
## Fitting the Data Set leaf node = 40
classifier40 = classifier40.fit(X2_train, y2_train)
y40_pred = classifier40.predict(X2_test)
## Creating confusion Matrix leaf node = 40 
cm40 = confusion_matrix(y2_test, y40_pred)


# Creating decision trees “minimum records per leaf node” values of  50
classifier50 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 50, random_state = 0)
## Fitting the Data Set leaf node = 50 
classifier50 = classifier50.fit(X2_train, y2_train)
y50_pred = classifier50.predict(X2_test)
## Creating confusion Matrix leaf node = 50 
cm50 = confusion_matrix(y2_test, y50_pred)

#Graphing trees
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

pip install graphviz
pip install pydotplus
pip install pyparsing
import pydotplus as pypl

# Graphing 5 leaves node 
dot_data5 = StringIO()
export_graphviz(classifier5, out_file=dot_data5,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data5.getvalue())  
Image(graph.create_png())
graph.write_png('5leavesNode.png')

# Graphing 15 leaves node 
dot_data15 = StringIO()
export_graphviz(classifier15, out_file=dot_data15,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data15.getvalue())  
Image(graph.create_png())
graph.write_png('15leavesNode.png')


# Graphing 25 leaves node 
dot_data25 = StringIO()
export_graphviz(classifier25, out_file=dot_data25,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data25.getvalue())  
Image(graph.create_png())
graph.write_png('25leavesNode.png')


# Graphing 40 leaves node 
dot_data40 = StringIO()
export_graphviz(classifier40, out_file=dot_data40,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data40.getvalue())  
Image(graph.create_png())
graph.write_png('40leavesNode.png')


# Graphing 50 leaves node 
dot_data50 = StringIO()
export_graphviz(classifier50, out_file=dot_data50,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data50.getvalue())  
Image(graph.create_png())
graph.write_png('50leavesNode.png')


from sklearn.metrics import classification_report
print(classification_report(y2_test, y5_pred))
print(classification_report(y2_test, y15_pred))
print(classification_report(y2_test, y25_pred))
print(classification_report(y2_test, y40_pred))
print(classification_report(y2_test, y50_pred))

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y2_test, y5_pred)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

print("The prediction accuracy is: ",classifier5.score(X2_test, y2_test)*100,"%")
print("The prediction accuracy is: ",classifier50.score(X2_test, y2_test)*100,"%")

### Work on it later 
# Plot Precision Recall 
import scipy as sp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve




disp = plot_precision_recall_curve(classifier5, X2_test, y2_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


#Question 2 

#Importing the data set-2
dataset3 = pd.read_csv('Biomechanical_Data_3Classes.csv')
X3 = dataset3.iloc[:, :-1].values
y3 = dataset3.iloc[:, 6:7].values

#Encoding Categorical Value 
labelencoder3 = LabelEncoder()
y3[:, 0] = labelencoder3.fit_transform(y3[:, 0])

# Spliiting the data set as trainin and test sets 
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 8/31, random_state = 0)
y3_train = y3_train.astype('int')
y3_test = y3_test.astype('int')


# Creating decision trees “minimum records per leaf node” values of 5
classifier5_3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 5, random_state = 0)
## Fitting the Data Set leaf node = 5 
classifier5_3 = classifier5_3.fit(X3_train, y3_train)
y5_pred_3 = classifier5_3.predict(X3_test)
##Creating confusion Matrix leaf node = 5 
cm5_3 = confusion_matrix(y3_test, y5_pred_3)


# Creating decision trees “minimum records per leaf node” values of 15
classifier15_3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 15, random_state = 0)
## Fitting the Data Set leaf node = 15 
classifier15_3 = classifier15_3.fit(X3_train, y3_train)
y15_pred_3 = classifier15_3.predict(X3_test)
## Creating confusion Matrix leaf node = 15 
cm15_3 = confusion_matrix(y3_test, y15_pred_3)


# Creating decision trees “minimum records per leaf node” values of  25
classifier25_3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 25, random_state = 0)
## Fitting the Data Set leaf node = 25 
classifier25_3 = classifier25_3.fit(X3_train, y3_train)
y25_pred_3 = classifier25_3.predict(X3_test)
## Creating confusion Matrix leaf node = 25 
cm25_3 = confusion_matrix(y3_test, y25_pred_3)


# Creating decision trees “minimum records per leaf node” values of  40
classifier40_3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 40, random_state = 0)
## Fitting the Data Set leaf node = 40
classifier40_3 = classifier40_3.fit(X3_train, y3_train)
y40_pred_3 = classifier40_3.predict(X3_test)
## Creating confusion Matrix leaf node = 40 
cm40_3 = confusion_matrix(y3_test, y40_pred_3)


# Creating decision trees “minimum records per leaf node” values of  50
classifier50_3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 50, random_state = 0)
## Fitting the Data Set leaf node = 50 
classifier50_3 = classifier50_3.fit(X3_train, y3_train)
y50_pred_3 = classifier50_3.predict(X3_test)
## Creating confusion Matrix leaf node = 50 
cm50_3 = confusion_matrix(y3_test, y50_pred_3)

#Graphing trees

# Graphing 5 leaves node 
dot_data5_3 = StringIO()
export_graphviz(classifier5_3, out_file=dot_data5_3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1', '2'])
graph = pypl.graph_from_dot_data(dot_data5_3.getvalue())  
Image(graph.create_png())
graph.write_png('5leavesNode_Data_3.png')

# Graphing 15 leaves node 
dot_data15_3 = StringIO()
export_graphviz(classifier15_3, out_file=dot_data15_3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1', '2'])
graph = pypl.graph_from_dot_data(dot_data15_3.getvalue())  
Image(graph.create_png())
graph.write_png('15leavesNode_Data_3.png')


# Graphing 25 leaves node 
dot_data25_3 = StringIO()
export_graphviz(classifier25_3, out_file=dot_data25_3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1', '2'])
graph = pypl.graph_from_dot_data(dot_data25_3.getvalue())  
Image(graph.create_png())
graph.write_png('25leavesNode_Data_3.png')


# Graphing 40 leaves node 
dot_data40_3 = StringIO()
export_graphviz(classifier40_3, out_file=dot_data40_3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1', '2'])
graph = pypl.graph_from_dot_data(dot_data40_3.getvalue())  
Image(graph.create_png())
graph.write_png('40leavesNode_Data_3.png')


# Graphing 50 leaves node 
dot_data50_3 = StringIO()
export_graphviz(classifier50_3, out_file=dot_data50_3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1', '2'])
graph = pypl.graph_from_dot_data(dot_data50_3.getvalue())  
Image(graph.create_png())
graph.write_png('50leavesNode_Data_3.png')


#Printing Accuracy, Precision Recall
print(classification_report(y3_test, y5_pred_3))
print(classification_report(y3_test, y15_pred_3))
print(classification_report(y3_test, y25_pred_3))
print(classification_report(y3_test, y40_pred_3))
print(classification_report(y3_test, y50_pred_3))

#Plotting the Recall and Precision

# Question 3 

#Importing the data set-2

dataset2_q3 = pd.read_csv('Biomechanical_Data_2Classes.csv')
dataset2_q3 = dataset2_q3.rename(columns = {'class':'diagnosed'})
dataset2_q3.loc[dataset2_q3.diagnosed != 'Normal', 'diagnosed'] = 1
dataset2_q3.loc[dataset2_q3.diagnosed == 'Normal', 'diagnosed'] = 0



#Finding the correlation 
corr = dataset2_q3.corr(method ='pearson') 
del dataset2_q3['sacral_slope']

#Splitting Data Set 
X2_q3 = dataset2_q3.iloc[:, :-1].values
y2_q3 = dataset2_q3.iloc[:, 5:6].values

# Spliiting the data set as trainin and test sets 
X2_q3train, X2_q3test, y2_q3train, y2_q3test = train_test_split(X2_q3, y2_q3, test_size = 8/31, random_state = 1234)



# Creating decision trees “minimum records per leaf node” values of 5
classifier5_q3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 5, random_state = 0)
## Fitting the Data Set leaf node = 5 
classifier5_q3 = classifier5_q3.fit(X2_q3train, y2_q3train)
y5_pred_q3 = classifier5_q3.predict(X2_q3test)
##Creating confusion Matrix leaf node = 5 
cm5_q3 = confusion_matrix(y2_q3test, y5_pred_q3)


# Creating decision trees “minimum records per leaf node” values of 15
classifier15_q3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 15, random_state = 0)
## Fitting the Data Set leaf node = 15 
classifier15_q3 = classifier15_q3.fit(X2_q3train, y2_q3train)
y15_pred_q3 = classifier15_q3.predict(X2_q3test)
## Creating confusion Matrix leaf node = 15 
cm15_q3 = confusion_matrix(y2_q3test, y15_pred_q3)


# Creating decision trees “minimum records per leaf node” values of  25
classifier25_q3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 25, random_state = 0)
## Fitting the Data Set leaf node = 25 
classifier25_q3 = classifier25_q3.fit(X2_q3train, y2_q3train)
y25_pred_q3 = classifier25_q3.predict(X2_q3test)
## Creating confusion Matrix leaf node = 25 
cm25_q3 = confusion_matrix(y2_q3test, y25_pred_q3)


# Creating decision trees “minimum records per leaf node” values of  40
classifier40_q3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 40, random_state = 0)
## Fitting the Data Set leaf node = 40
classifier40_q3 = classifier40_q3.fit(X2_q3train, y2_q3train)
y40_pred_q3 = classifier40_q3.predict(X2_q3test)
## Creating confusion Matrix leaf node = 40 
cm40_q3 = confusion_matrix(y2_q3test, y40_pred_q3)


# Creating decision trees “minimum records per leaf node” values of  50
classifier50_q3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 50, random_state = 0)
## Fitting the Data Set leaf node = 50 
classifier50_q3 = classifier50_q3.fit(X2_q3train, y2_q3train)
y50_pred_q3 = classifier50_q3.predict(X2_q3test)
## Creating confusion Matrix leaf node = 50 
cm50_q3 = confusion_matrix(y2_q3test, y50_pred_q3)

#Graphing trees

# Graphing 5 leaves node 
dot_data5_q3 = StringIO()
export_graphviz(classifier5_q3, out_file=dot_data5_q3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['1', '0'])
graph = pypl.graph_from_dot_data(dot_data5_q3.getvalue())  
Image(graph.create_png())
graph.write_png('5leavesNodeq3.png')

# Graphing 15 leaves node 
dot_data15_q3 = StringIO()
export_graphviz(classifier15_q3, out_file=dot_data15_q3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data15_q3.getvalue())  
Image(graph.create_png())
graph.write_png('15leavesNodeq3.png')


# Graphing 25 leaves node 
dot_data25_q3 = StringIO()
export_graphviz(classifier25_q3, out_file=dot_data25_q3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data25_q3.getvalue())  
Image(graph.create_png())
graph.write_png('25leavesNodeq3.png')


# Graphing 40 leaves node 
dot_data40_q3 = StringIO()
export_graphviz(classifier40_q3, out_file=dot_data40_q3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data40_q3.getvalue())  
Image(graph.create_png())
graph.write_png('40leavesNodeq3.png')


# Graphing 50 leaves node 
dot_data50_q3 = StringIO()
export_graphviz(classifier50_q3, out_file=dot_data50_q3,  
                filled=True, rounded=True,
                special_characters=True, class_names = ['0', '1'])
graph = pypl.graph_from_dot_data(dot_data50_q3.getvalue())  
Image(graph.create_png())
graph.write_png('50leavesNodeq3.png')





from sklearn.metrics import classification_report
print(classification_report(y2_q3test, y5_pred_q3))
print(classification_report(y2_q3test, y15_pred_q3))
print(classification_report(y2_q3test, y25_pred_q3))
print(classification_report(y2_q3test, y40_pred_q3))
print(classification_report(y2_q3test, y50_pred_q3))

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y2_q3test, y5_pred_q3)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

print("The prediction accuracy is: ",classifier5.score(X2_test, y2_test)*100,"%")
print("The prediction accuracy is: ",classifier50.score(X2_test, y2_test)*100,"%")

### Work on it later 
# Plot Precision Recall 
import scipy as sp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve




disp = plot_precision_recall_curve(classifier5, X2_test, y2_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


