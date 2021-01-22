#Import Libraries
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Import fundamentals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

kdd_test = pd.read_csv("C:/Users/Tim/python_projects/CybersecurityModel/NSL-KDD Dataset Test and Train/kdd_test.csv")
kdd_train = pd.read_csv("C:/Users/Tim/python_projects/CybersecurityModel/NSL-KDD Dataset Test and Train/kdd_train.csv")

#define function for classifying players based on points
def category(row):
    if row['labels'] == "normal":
        val = 'Normal'
    else:
        val = 'Anomaly'
    return val

#create new column 'Good' using the function above
kdd_train["category"] = kdd_train.apply(category, axis=1)

# sns.countplot(x='category',data=kdd_train, palette='hls')
# plt.xlabel('Type of Traffic')
# plt.ylabel('Total')
# plt.title('Number of Each form of Traffic')
# plt.xticks(rotation=90)
# plt.show()

#Label Encode Part 1
le = LabelEncoder()
kdd_train_cat = kdd_train[["flag","category"]]

# print(kdd_train_cat.groupby('flag').describe())

kdd_train_cat_enc = kdd_train_cat.apply(le.fit_transform)

kdd_train_temp = kdd_train.drop("flag", axis=1)
kdd_train_temp = kdd_train_temp.drop("category", axis=1)
kdd_train_temp = kdd_train_temp.drop("num_outbound_cmds", axis=1)
kdd_train_temp = kdd_train_temp.drop("labels", axis=1)

kdd_temp_final = kdd_train_cat_enc.join(kdd_train_temp)

#Correlation
print(kdd_temp_final)
cor = kdd_temp_final.corr()
print(cor)

#Correlation with output variable
cor_target = cor["category"]
#Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
print(relevant_features)

plt.figure(figsize=(10,6))
sns.heatmap(cor, annot=True)
plt.show()

#Label Encode Part 2
le = LabelEncoder()
kdd_train_cat = kdd_train[["flag","category"]]

kdd_train_cat_enc = kdd_train_cat.apply(le.fit_transform)

kdd_train_nominal = kdd_train[["logged_in", "same_srv_rate", "dst_host_srv_count", "dst_host_same_srv_rate"]]

kdd_train_final = kdd_train_cat_enc.join(kdd_train_nominal)

# from pandas.plotting import scatter_matrix
# attributes = ["flag","logged_in","same_srv_rate","dst_host_srv_count","dst_host_same_srv_rate","category"]
# scatter_matrix(kdd_train[attributes], figsize=(12,8))
# plt.show()

sc = StandardScaler()
X = kdd_train_final.drop("category",1)
y = kdd_train_final["category"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#print the length of both test and train set
print('Length of X_train : ', len(X_train), '\n Length of y_train : ', len(y_train))
print('Length of X_test : ', len(X_test), '\n Length of y_test : ', len(y_test))

#Normalise Data via Standard SCaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#__________________________________________________________
#Modeling
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

accuracy = cross_val_score(clf, X_test, y_test, cv=10, scoring='accuracy').max()
print("Accuracy of Model:",accuracy)
print(accuracy.mean())

# Predict the Labels/Categories
y_pred2 = clf.predict(X_test)

print("This is the Predict Input")
print(X_test)

# Print the Confusion Matrix
cm = confusion_matrix(y_test, y_pred2)
print("Confusion Matrix\n")
print(cm)
sns.heatmap(cm, annot=True, fmt='d',cmap='rocket')
plt.show()

# Print the Classification Report
cr = classification_report(y_test, y_pred2)
print("\n\nClassification Report\n")
print(cr)

import streamlit as st

st.write("""
# IDS Thingy
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    flag = st.slider('flag', 0, 11, value=2,step=1)
    logged_in = st.slider('logged_in', 0, 11, value=2,step=1)
    same_srv_rate = st.slider('same_srv_rate', 0, 11, value=2,step=1)
    dst_host_srv_count = st.slider('dst_host_srv_count', 0, 11, value=2,step=1)
    dst_host_same_srv_rate = st.slider('dst_host_same_srv_rate', 0, 11, value=2,step=1)
    data = {'flag' : flag,
            'logged_in': logged_in,
            'same_srv_rate': same_srv_rate,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': dst_host_same_srv_rate}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')

traffic_target = np.array(['Normal','Anomaly'])
st.write(traffic_target)

st.subheader('Prediction')
st.write(traffic_target[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)