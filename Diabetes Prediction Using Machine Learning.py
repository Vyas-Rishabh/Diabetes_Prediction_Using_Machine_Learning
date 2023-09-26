#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction Using Machine Learning

# ## Overview
# ## we will be predicting that whether the patient has diabetes or not on the basis of the features
# ## we will provide to our machine learning model, and for that, we will be using the famous Pima Indians Diabetes Database.

# Data analysis: Here one will get to know about how the data analysis part is done in a data science life cycle. Exploratory data analysis: EDA is one of the most important steps in the data science
# project life cycle and here one will need to know that how to make inferences from the visualizations and data analysis Model building: Here we will be using 4 ML models and then we will choose the
# best performing model. Saving model: Saving the best model using pickle to make the prediction from real data.

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Here we will be reading the dataset which is in the CSV format

diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df.head()


# ## Exploratory Data Analysis (EDA)

# In[3]:


diabetes_df.columns


# In[4]:


# Information about the dataset

diabetes_df.info()


# In[5]:


# To know the more about dataset

diabetes_df.describe()


# In[6]:


# To know more about the dataset with transpose - here T is for the transpose

diabetes_df.describe().T


# In[7]:


# Now let's check that if our dataset have null values or not

diabetes_df.isnull().sum()


# <b>Here from the above code we first checked that is there any null values from the
# IsNull() function then we are going to take the sum of all those missing values from the
# sum() function and the inference we now get is that there are no missing values but
# that is actually not a true story as in this particular dataset all the missing values were
# given the 0 as a value which is not good for the authenticity of the dataset. Hence we
# will first replace the 0 value with the NAN value then start the imputation process.</b>

# In[8]:


diabetes_df_copy = diabetes_df.copy(deep = True)
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
                            diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NAN)


# In[9]:


# Showing the count of NaNs
print(diabetes_df_copy.isnull().sum())


# #### As mentioned above that now we will be replacing the zeros with the NAN values so that we can impute it later to maintain the authenticity of the dataset as well as trying to have a better Imputation approach i.e to apply mean values of each column to the null values of the respective columns.

# ## Data Visualization

# In[10]:


#Plotting the data distrubution plots before removing null values

p = diabetes_df.hist(figsize=(20,20))


# #### Inference: So here we have seen the distribution of each features whether it is dependent data or independent data and one thing which could always strike that why do we need to see the distribution of data? So the answer is simple it is the best way to start the analysis of the dataset as it shows the occurrence of every kind of value in the graphical structure which in turn lets us know the range of the data.

# In[11]:


#Now we will be imputing the mean value of the column to each missing value of that particular column.

diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace=True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace=True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace=True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace=True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace=True)


# In[12]:


# Plotting the distrubution after  removing NAN values.

p = diabetes_df_copy.hist(figsize=(20,20))


# Inference: Here we are again using the hist plot to see the distribution of the dataset but this time we are using this visualization to see the changes that we can see after those null values are removed from the dataset and we can clearly see the difference for example – In age column after removal of the null values, we can see that there is a spike at the range of 50 to 100 which is quite logical as well.

# In[13]:


# Plotting Null Count Analysis Plot

p = msno.bar(diabetes_df)


# #### Inference: Now in the above graph also we can clearly see that there are no null values in the dataset.

# In[19]:


#Now, Let's check that how well our outcome column is balanced

color_wheel = {1: "#0392cf", 2: "#7bc043"}
#colors = diabetes_df["Outcome"]
print(diabetes_df.Outcome.value_counts())
p = diabetes_df.Outcome.value_counts().plot(kind='bar')


# ### Here from the above visualization it is clearly visible that our dataset is completely imbalanced in fact the number of patients who are diabetic is half of the patients who are non-diabetic.

# In[22]:


plt.subplot(121), sns.distplot(diabetes_df['Insulin'])
plt.subplot(122), diabetes_df['Insulin'].plot.box(figsize=(16,5))
plt.show()


# #### That’s how Distplot can be helpful where one will able to see the distribution of the data as well as with the help of boxplot one can see the outliers in that column and other information too which can be derived by the box and whiskers plot.

# # Correlation between all the features

# In[23]:


#Correlation between all the features before clearning

plt.figure(figsize=(8,5))
# seaborn has an easy method to showcase heatmap
p = sns.heatmap(diabetes_df.corr(), annot=True, cmap='RdYlGn')


# # Scalling the Data

# In[24]:


# Before scaling down the data let's have a look into it

diabetes_df_copy.head()


# In[25]:


#After Standard scaling

sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 
'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()


# #### That’s how our dataset will be looking like when it is scaled down or we can see every value now is on the same scale which will help our ML model to give a better result.

# # Model Building

# In[26]:


#Splitting the dataset

X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']


# In[28]:


#Now we will split the data into training and testing data using the train test split function

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=7)


# # Random Forest

# In[31]:


# Building the model using RandomForest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)


# In[32]:


#Now after building the model let’s check the accuracy of the model on the training dataset.

rfc_train = rfc.predict(X_train)
from sklearn import metrics

print("Accuracy_Score =", format(metrics.accuracy_score(y_train, rfc_train)))


# #### So here we can see that on the training dataset our model is overfitted. Getting the accuracy score for Random Forest

# In[33]:


from sklearn import metrics

predictions = rfc.predict(X_test)
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))


# In[34]:


#Classification report and confusion matrix of random forest model
# Train the model
clf = RandomForestClassifier(random_state=23)
clf.fit(X_train, y_train)
 
# preduction
y_pred = clf.predict(X_test)
 
# compute the confusion matrix
cm = confusion_matrix(y_test,y_pred)
 
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()
 
 
# Finding precision and recall
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy   :", accuracy)


# In[35]:


# Finding precision and recall
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy   :", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision :", precision)
recall = recall_score(y_test, y_pred)
print("Recall    :", recall)
F1_score = f1_score(y_test, y_pred)
print("F1-score  :", F1_score)


# # Decision Tree

# In[36]:


#Building the model using DecisionTree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

#Now we will be making the predictions on the testing data directly as it is of more importance.


# In[37]:


#Getting the accuracy score for Decision Tree

from sklearn import metrics

predictions = dtree.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions)))


# In[38]:


#Classification report and confusion matrix of the decision tree model

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))


# # XgBoost classifier

# In[41]:


#Building model using XGBoost

from xgboost import XGBClassifier

xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)


# In[42]:


#Now we will be making the predictions on the testing data directly as it is of more importance.
#Getting the accuracy score for the XgBoost classifier

from sklearn import metrics

xgb_pred = xgb_model.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test, xgb_pred)))


# In[43]:


#Classification report and confusion matrix of the XgBoost classifier


# In[44]:


#Support Vector Machine (SVM)

#Building the model using Support Vector Machine (SVM)

from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)

#Prediction from support vector machine model on the testing data

svc_pred = svc_model.predict(X_test)

#Accuracy score for SVM

from sklearn import metrics

print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred)))


# In[45]:


# Classification report and confusion matrix of the SVM classifier

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test,svc_pred))


# # The Conclusion from Model Building
# Therefore Random forest is the best model for this prediction since it has an accuracy_score of 0.76.

# # Feature Importance
# Knowing about the feature importance is quite necessary as it shows that how much weightage each feature provides in the model building phase.

# In[46]:


#Getting feature importances

rfc.feature_importances_


# #### From the above output, it is not much clear that which feature is important for that reason we will now make a visualization of the same.

# In[47]:


#Plotting feature importances

(pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh'))


# ##### Here from the above graph, it is clearly visible that Glucose as a feature is the most important in this dataset.

# # Saving Model – Random Forest

# In[48]:


import pickle

# Firstly we will be using the dump() function to save the model using pickle
saved_model = pickle.dumps(rfc)

# Then we will be loading that saved model
rfc_from_pickle = pickle.loads(saved_model)

# lastly, after loading that model we will use this to make predictions
rfc_from_pickle.predict(X_test)


# ##### Now for the last time, I’ll be looking at the head and tail of the dataset so that we can take any random set of features from both the head and tail of the data to test that if our model is good enough to give the right prediction.

# In[49]:


diabetes_df.head()


# In[50]:


diabetes_df.tail()


# In[51]:


#Putting data points in the model will either return 0 or 1 i.e. person suffering from diabetes or not.

rfc.predict([[0,137,40,35,168,43.1,2.228,33]]) #4th patient


# In[52]:


#Another one

rfc.predict([[10,101,76,48,180,32.9,0.171,63]])  # 763 th patient


# # Conclusion
# After using all these patient records, we are able to build a machine learning model (random forest – best one) to accurately
# 
# predict whether or not the patients in the dataset have diabetes or not along with that we were able to draw some insights from
# 
# the data via data analysis and visualization.
