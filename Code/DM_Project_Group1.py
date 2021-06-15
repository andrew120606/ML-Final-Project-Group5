# %%%%%%%%%%%%% Data Mining%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Zongzhu Li------>Email: zongzhuli@gwmail.gwu.edu
#
# Renping Ge
# %%%%%%%%%%%%% Date:
# April- 15 - 2021
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Project  %%%%%%%%%%%%%%%%%
#%%-----------------------------------------------------------------------
## Importing the required packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#%%-----------------------------------------------------------------------
## Importing dataset and check dataset information
# read data as panda dataframe
data = pd.read_csv("C:\\Andrew120606\\GWU_DATA_SCIENCE\\courses\\IntroductiontoDataMining\\final project\\aug_train.csv")

# printing the dataset obseravtions
print("Dataset first few rows:\n ")
print(data.head())

# printing the dataset struture
print("Dataset info:\n ")
print(data.info())

# printing the the dataset summary
print(data.describe(include='all'))

#%%-----------------------------------------------------------------------
## Features and Target
print("Features and Target:\n"
      "enrollee_id: Unique ID for enrollee.\n"
      "city: City code.\n"
      "citydevelopmentindex: Developement index of the city (scaled).\n"
      "gender: Gender of enrolee.\n"
      "relevent_experience: Relevent experience of enrolee.\n"
      "enrolled_university: Type of University course enrolled if any.\n"
      "education_level: Education level of enrolee.\n"
      "major_discipline: Education major discipline of enrolee.\n"
      "experience: Enrolee total experience in years.\n"
      "company_size: No of employees in current employer's company.\n"
      "company_type: Type of current employer.\n"
      "last_new_job: Difference in years between previous job and current job.\n"
      "training_hours: training hours completed.\n"
      "target: 0 – Not looking for job change, 1 – Looking for a job change.")

#%%-----------------------------------------------------------------------
## Preparation for EDA
# remove unrelated columns
data = data.drop(["enrollee_id","city"],axis=1)
print(data.info())

# change some values to be understood easily
data["company_size"].unique()
for i in range(len(data.index)):
    if data['company_size'][i] == '10/49':
        data['company_size'][i] = '10-49'

data["experience"].unique()
for i in range(len(data.index)):
    if data['experience'][i] == '>20':
        data['experience'][i] = '21'
    elif data['experience'][i] == '<1':
       data['experience'][i] = '0'

data["last_new_job"].unique()
for i in range(len(data.index)):
    if data['last_new_job'][i] == '>4':
        data['last_new_job'][i] = '5'
    elif data['last_new_job'][i] == 'never':
        data['last_new_job'][i] = '0'

retarget = {0.0: 'Not looking for job change',
           1.0: 'Looking for job change'}
data['target'] = data['target'].map(retarget)

#%%-----------------------------------------------------------------------
## EDA
# show counts for target
target = data.groupby('target').agg({'target': 'count'}).rename(columns = {'target': 'count'}).reset_index()
a = sns.barplot(data = target,x = target['target'], y = target['count'])
for p in a.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/len(data.target))
    x = p.get_x() + p.get_width() / 2 -0.1
    y = p.get_y() + p.get_height()
    a.annotate(percentage, (x, y), size = 12)
plt.title('target', size = 16)
plt.show()

# Distribution of job change by gender
gender_df = data.groupby(['gender', 'target']).agg({'target': 'count'}).rename(columns = {'target': 'count'}).reset_index()
# genderdf_agg = genderdf.groupby(['gender'])['count'].sum().reset_index()
# genderdf2 = genderdf.merge(genderdf_agg, on='gender', how='left')
# genderdf2['percentage']=round(genderdf2.count_x/genderdf2.count_y * 100,1)
b = sns.barplot(data = gender_df, x = gender_df['gender'], y = gender_df['count'],hue = gender_df['target'])

patch_height = [p.get_height() for p in b.patches]
patch = [p for p in b.patches]
for i in range(gender_df["gender"].unique().size):
    total = gender_df.groupby(['gender'])['count'].sum().values[i]
    for j in range(gender_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * gender_df["gender"].unique().size+i)]/total)
        x = patch[(j * gender_df["gender"].unique().size+i)].get_x() + patch[(j * gender_df["gender"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * gender_df["gender"].unique().size+i)].get_y() + patch[(j * gender_df["gender"].unique().size+i)].get_height()
        b.annotate(percentage, (x, y), size = 12)
plt.title('gender', size = 16)
plt.show()

# Distribution of job change by relevent_experience
rel_exp_df = data.groupby(["relevent_experience", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
c = sns.barplot(data = rel_exp_df, x = rel_exp_df["relevent_experience"], y = rel_exp_df["count"],hue = rel_exp_df["target"])
patch_height = [p.get_height() for p in c.patches]
patch = [p for p in c.patches]
for i in range(rel_exp_df["relevent_experience"].unique().size):
    total = rel_exp_df.groupby(["relevent_experience"])["count"].sum().values[i]
    for j in range(rel_exp_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * rel_exp_df["relevent_experience"].unique().size+i)]/total)
        x = patch[(j * rel_exp_df["relevent_experience"].unique().size+i)].get_x() + patch[(j * rel_exp_df["relevent_experience"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * rel_exp_df["relevent_experience"].unique().size+i)].get_y() + patch[(j * rel_exp_df["relevent_experience"].unique().size+i)].get_height()
        c.annotate(percentage, (x, y), size = 12)
plt.title("relevent_experience", size = 16)
plt.show()

# Distribution of job change by enrolled_university
enr_uni_df = data.groupby(["enrolled_university", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
d = sns.barplot(data = enr_uni_df, x = enr_uni_df["enrolled_university"], y = enr_uni_df["count"],hue = enr_uni_df["target"])
patch_height = [p.get_height() for p in d.patches]
patch = [p for p in d.patches]
for i in range(enr_uni_df["enrolled_university"].unique().size):
    total = enr_uni_df.groupby(["enrolled_university"])["count"].sum().values[i]
    for j in range(enr_uni_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * enr_uni_df["enrolled_university"].unique().size+i)]/total)
        x = patch[(j * enr_uni_df["enrolled_university"].unique().size+i)].get_x() + patch[(j * enr_uni_df["enrolled_university"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * enr_uni_df["enrolled_university"].unique().size+i)].get_y() + patch[(j * enr_uni_df["enrolled_university"].unique().size+i)].get_height()
        d.annotate(percentage, (x, y), size = 12)
plt.title("enrolled_university", size = 16)
plt.show()

# Distribution of job change by education_level
edu_lev_df = data.groupby(["education_level", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
e = sns.barplot(data = edu_lev_df, x = edu_lev_df["education_level"], y = edu_lev_df["count"],hue = edu_lev_df["target"])
patch_height = [p.get_height() for p in e.patches]
patch = [p for p in e.patches]
for i in range(edu_lev_df["education_level"].unique().size):
    total = edu_lev_df.groupby(["education_level"])["count"].sum().values[i]
    for j in range(edu_lev_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * edu_lev_df["education_level"].unique().size+i)]/total)
        x = patch[(j * edu_lev_df["education_level"].unique().size+i)].get_x() + patch[(j * edu_lev_df["education_level"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * edu_lev_df["education_level"].unique().size+i)].get_y() + patch[(j * edu_lev_df["education_level"].unique().size+i)].get_height()
        e.annotate(percentage, (x, y), size = 12)
plt.title("education_level", size = 16)
plt.show()

# Distribution of job change by major_discipline
maj_disci_df = data.groupby(["major_discipline", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
f = sns.barplot(data = maj_disci_df, x = maj_disci_df["major_discipline"], y = maj_disci_df["count"],hue = maj_disci_df["target"])
patch_height = [p.get_height() for p in f.patches]
patch = [p for p in f.patches]
for i in range(maj_disci_df["major_discipline"].unique().size):
    total = maj_disci_df.groupby(["major_discipline"])["count"].sum().values[i]
    for j in range(maj_disci_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * maj_disci_df["major_discipline"].unique().size+i)]/total)
        x = patch[(j * maj_disci_df["major_discipline"].unique().size+i)].get_x() + patch[(j * maj_disci_df["major_discipline"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * maj_disci_df["major_discipline"].unique().size+i)].get_y() + patch[(j * maj_disci_df["major_discipline"].unique().size+i)].get_height()
        f.annotate(percentage, (x, y), size = 12)
plt.title("major_discipline", size = 16)
plt.show()

# Distribution of job change by company_type
com_typ_df = data.groupby(["company_type", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
g = sns.barplot(data = com_typ_df, x = com_typ_df["company_type"], y = com_typ_df["count"],hue = com_typ_df["target"])
patch_height = [p.get_height() for p in g.patches]
patch = [p for p in g.patches]
for i in range(com_typ_df["company_type"].unique().size):
    total = com_typ_df.groupby(["company_type"])["count"].sum().values[i]
    for j in range(com_typ_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * com_typ_df["company_type"].unique().size+i)]/total)
        x = patch[(j * com_typ_df["company_type"].unique().size+i)].get_x() + patch[(j * com_typ_df["company_type"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * com_typ_df["company_type"].unique().size+i)].get_y() + patch[(j * com_typ_df["company_type"].unique().size+i)].get_height()
        g.annotate(percentage, (x, y), size = 12)
plt.title("company_type", size = 16)
plt.show()

# Distribution of job change by experience
exper_df = data.groupby(["experience", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
exper_df["experience"] = exper_df["experience"].astype("int")
#order = ("0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21")
h = sns.barplot(data = exper_df, x = exper_df["experience"], y = exper_df["count"],hue = exper_df["target"])
patch_height = [p.get_height() for p in h.patches]
patch = [p for p in h.patches]
for i in range(exper_df["experience"].unique().size):
    total = exper_df.groupby(["experience"])["count"].sum().values[i]
    for j in range(exper_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * exper_df["experience"].unique().size+i)]/total)
        x = patch[(j * exper_df["experience"].unique().size+i)].get_x() + patch[(j * exper_df["experience"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * exper_df["experience"].unique().size+i)].get_y() + patch[(j * exper_df["experience"].unique().size+i)].get_height()
        h.annotate(percentage, (x, y), size = 8)
plt.title("experience", size = 16)
plt.show()

# Distribution of job change by company_size
com_siz_df = data.groupby(["company_size", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
k = sns.barplot(data = com_siz_df, x = com_siz_df["company_size"], y = com_siz_df["count"],hue = com_siz_df["target"])
patch_height = [p.get_height() for p in k.patches]
patch = [p for p in k.patches]
for i in range(com_siz_df["company_size"].unique().size):
    total = com_siz_df.groupby(["company_size"])["count"].sum().values[i]
    for j in range(com_siz_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * com_siz_df["company_size"].unique().size+i)]/total)
        x = patch[(j * com_siz_df["company_size"].unique().size+i)].get_x() + patch[(j * com_siz_df["company_size"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * com_siz_df["company_size"].unique().size+i)].get_y() + patch[(j * com_siz_df["company_size"].unique().size+i)].get_height()
        k.annotate(percentage, (x, y), size = 12)
plt.legend(loc = 'upper left')
plt.title("company_size", size = 16)
plt.show()

# Distribution of job change by last_new_job
last_new_df = data.groupby(["last_new_job", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
l = sns.barplot(data = last_new_df, x = last_new_df["last_new_job"], y = last_new_df["count"],hue = last_new_df["target"])
patch_height = [p.get_height() for p in l.patches]
patch = [p for p in l.patches]
for i in range(last_new_df["last_new_job"].unique().size):
    total = last_new_df.groupby(["last_new_job"])["count"].sum().values[i]
    for j in range(last_new_df["target"].unique().size):
        percentage = '{:.1f}%'.format(100 * patch_height[(j * last_new_df["last_new_job"].unique().size+i)]/total)
        x = patch[(j * last_new_df["last_new_job"].unique().size+i)].get_x() + patch[(j * last_new_df["last_new_job"].unique().size+i)].get_width() / 2 -0.1
        y = patch[(j * last_new_df["last_new_job"].unique().size+i)].get_y() + patch[(j * last_new_df["last_new_job"].unique().size+i)].get_height()
        l.annotate(percentage, (x, y), size = 12)
plt.title("last_new_job", size = 16)
plt.show()

# Distribution of job change by training_hours
sns.kdeplot(data.query('target == "Looking for job change"')['training_hours'], color = 'blue', shade = False, label = 'Looking for job change', alpha = 0.5)
sns.kdeplot(data.query('target == "Not looking for job change"')['training_hours'], color = 'red', shade = False, label = 'Not looking for job change', alpha = 0.5)

plt.xlabel('training_hours')
plt.ylabel('Density')
plt.yticks([])
plt.legend(loc = 'upper right')
plt.title('training_hours', size = 16)
plt.show()

# tra_hrs_df = data.groupby(["training_hours", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
# sns.displot(data, x="training_hours",hue="target", kind="kde")
# #plt.legend(loc = 'best')
# plt.show()
# tra_hrs_df = data.groupby(["training_hours", "target"]).agg({"target": "count"}).rename(columns = {"target": "count"}).reset_index()
# m = sns.barplot(data = tra_hrs_df, x = tra_hrs_df["training_hours"], y = tra_hrs_df["count"],hue = tra_hrs_df["target"])
# patch_height = [p.get_height() for p in m.patches]
# patch = [p for p in m.patches]
# for i in range(tra_hrs_df["training_hours"].unique().size):
#     total = tra_hrs_df.groupby(["training_hours"])["count"].sum().values[i]
#     for j in range(tra_hrs_df["target"].unique().size):
#         percentage = '{:.1f}%'.format(100 * patch_height[(j * tra_hrs_df["training_hours"].unique().size+i)]/total)
#         x = patch[(j * tra_hrs_df["training_hours"].unique().size+i)].get_x() + patch[(j * tra_hrs_df["training_hours"].unique().size+i)].get_width() / 2 -0.1
#         y = patch[(j * tra_hrs_df["training_hours"].unique().size+i)].get_y() + patch[(j * tra_hrs_df["training_hours"].unique().size+i)].get_height()
#         m.annotate(percentage, (x, y), size = 12)
# plt.title("training_hours", size = 16)
# plt.show()

# Distribution of job change by city_development_index
sns.kdeplot(data.query('target == "Looking for job change"')['city_development_index'], color = 'blue', shade = False, label = 'Looking for job change', alpha = 0.5)
sns.kdeplot(data.query('target == "Not looking for job change"')['city_development_index'], color = 'red', shade = False, label = 'Not looking for job change', alpha = 0.5)

plt.xlabel('city_development_index')
plt.ylabel('Density')
plt.yticks([])
plt.legend(loc = 'upper left')
plt.title('city_development_index', size = 16)
plt.show()

#%%-----------------------------------------------------------------------
## Summary for EDA
print("1. The rate of looking for a new job for people with no relevant experience  is a little  higher than that of people with relevent experience.\n"
      "2. People who took the full time course are more likely to look for a new job compared to others.\n"
      "3. People with graduate education level are more inclined to look for a new job than people with other education level.\n"
      "4. People with different major discipline shows comparable rate of looking for a new job.\n"
      "5. People with less working experiences are more likely to look for a new job.\n"
      "6. People working in the company with size of 10-49 are more inclined to look for a new job than people working in other size company.\n"
      "7. The difference years between working in current job and last job shows a role in the desire to change job, the difference of 1 year and zero year shows a significant higher rate of looking for a new job.\n"
      "8. The City Development Index shows a important role in the desire to change job:\n   In the cities with lower City Development Index, more people is likely to look for a new job, which means that those cities have more oppotunities."
      "9. People with different training hours shows comparable rate of looking for a new job.\n")

#%%-----------------------------------------------------------------------
## Preparation for Data Modeling
# remap target variable
retarget2 = {'Not looking for job change': 0,
             'Looking for job change': 1}
data['target'] = data['target'].map(retarget2)

# convert the necessary columns to a numeric format
data['experience'] = data['experience'].astype(np.float).astype("Int32")
data['last_new_job'] = data['last_new_job'].astype(np.float).astype("Int32")

# correlation between features with numerical data
cor=data.corr()
sns.heatmap(cor, annot=True)
plt.title('Correlation between features')
plt.show()

# specify the predictors and target variable
X = data.drop(["target"],axis = 1)
y = data["target"]

# X = data.values[:, 0:11]
# y = data.values[:, 11]

# fill na
print("Sum of NULL values in each column. ")
print(data.isnull().sum())
X['experience'] = X['experience'].astype('float64').fillna(X['experience'].mean())
X['last_new_job'] = X['last_new_job'].astype('float64').fillna(X['last_new_job'].mean())
X['training_hours'] = X['training_hours'].astype('float64').fillna(X['training_hours'].mean())

# # standerization and centralization
# X.dropna(how='any')
sc = StandardScaler()
X["city_development_index"] = sc.fit_transform(X["city_development_index"].values.reshape(-1,1))
X["experience"] = sc.fit_transform(X["experience"].values.reshape(-1,1))
X["last_new_job"] = sc.fit_transform(X["last_new_job"].values.reshape(-1,1))
X["training_hours"] = sc.fit_transform(X["training_hours"].values.reshape(-1,1))

# encoding categorical features with OneHotEncoder()
columns_categorical = ["gender","relevent_experience","enrolled_university","education_level","major_discipline","company_size","company_type"]
columns_numerical = ["city_development_index","experience","last_new_job","training_hours"]

X = pd.get_dummies(X, columns = columns_categorical)

# label target variable
le = LabelEncoder()
y = le.fit_transform(y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2000)

#%%-----------------------------------------------------------------------
## Modeling
# perform training with random forest with all columns
# specify random forest classifier and perform training
clf = RandomForestClassifier(n_estimators=90)
clf.fit(X_train, y_train)

#%%-----------------
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X.columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()
#%%-----------------------------------------------------------------------
## Make predictions
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

#%%-----------------------------------------------------------------------
## Model evaluation
# report
print(classification_report(y_test,y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("ROC_AUC:", roc_auc_score(y_test,y_pred_score[:,-1]) * 100)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data["target"].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)

plt.tight_layout()
plt.show()
#%%-----------------------------------------------------------------------
# Plot ROC Area Under Curve
y_pred_score = clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_score[:,-1])
auc = roc_auc_score(y_test, y_pred_score[:,-1])
#print(fpr)
#print(tpr)
#print(auc)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


#%%-----------------------------------------------------------------------
# KNN
clf_KNN = KNeighborsClassifier(n_neighbors=5)
clf_KNN.fit(X_train, y_train)
#make prediction
y_pred0 = clf_KNN.predict(X_test)
y_pred_score0 = clf_KNN.predict_proba(X_test)

print(classification_report(y_test,y_pred0))
print("Accuracy:", accuracy_score(y_test, y_pred0) * 100)
print("ROC_AUC:", roc_auc_score(y_test,y_pred_score0[:,-1]) * 100)

#%%-----------------------------------------------------------------------
# calculate metrics

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_KNN))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_KNN) * 100)
print("\n")

#%%------------------------------------------------------------------------------------------------------------------------
# boosting model
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

hgbc = HistGradientBoostingClassifier(random_state=42)
hgbc.fit(X_train, y_train)

y_valid_pred = hgbc.predict(X_test)
y_valid_pred_score = hgbc.predict_proba(X_test)

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_valid_pred, average='micro')

print(classification_report(y_test,y_valid_pred))
print("Accuracy:", accuracy_score(y_test, y_valid_pred) * 100)
print("ROC_AUC : ", roc_auc_score(y_test,y_valid_pred_score[:,1]) * 100)
# confusion matrix
conf_matrix = confusion_matrix(y_test, y_valid_pred)
df_cm = pd.DataFrame(conf_matrix, index=['0','1'], columns=['0','1'] )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()
# Plot ROC Area Under Curve
fpr, tpr, _ = roc_curve(y_test, y_valid_pred_score[:,-1])
auc = roc_auc_score(y_test, y_valid_pred_score[:,-1])
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

#%%------------------------------------------------------------------------------------
## Modeling
### logistic regression
from sklearn.linear_model import LogisticRegression
logt = LogisticRegression()
logt.fit(X_train,y_train)
y_valid_pred2 = logt.predict(X_test)
y_valid_pred2_score = logt.predict_proba(X_test)
print(classification_report(y_test,y_valid_pred2))
print("Accuracy : ", accuracy_score(y_test, y_valid_pred2) * 100)
print("ROC_AUC : ", roc_auc_score(y_test,y_valid_pred2_score[:,1]) * 100)
### confusion matrix
conf_matrix = confusion_matrix(y_test, y_valid_pred2)

df_cm = pd.DataFrame(conf_matrix, index=['0','1'], columns=['0','1'] )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%------------------------------------------------------------------
# Plot ROC Area Under Curve
fpr, tpr, _ = roc_curve(y_test, y_valid_pred2_score[:,-1])
auc = roc_auc_score(y_test, y_valid_pred2_score[:,-1])

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()
