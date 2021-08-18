#libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly
import plotlywidget
import plotly.express as px
pio.renderers.default = "png"


#read the dataset
main_dataset = pd.read_csv("water_potability.csv")

#made a copy version of the dataset
main_dataset_copy = main_dataset.copy()
main_dataset_copy2 = main_dataset.copy()

#first 5 data 
first5 = main_dataset_copy.head()

#last 5 data
last5 = main_dataset_copy.tail() 

#getting to know about the dataset
print(main_dataset.shape)

#column names
print(main_dataset.columns)

#column types
main_dataset.dtypes

#overall overview
overall_dataset = main_dataset.describe()

#information
print(main_dataset.info())

#total nun values
null_values = main_dataset.isnull().sum()

#heatmap of the null values for data visualization
sns.heatmap(main_dataset.isnull())

#cor-relation
correlation_dataset = main_dataset.corr()

#cor-relation heatmap
plt.figure(figsize = (10,8))
sns.heatmap(main_dataset.corr(), annot = True, cmap = plt.cm.CMRmap_r)

#list of highest cor-relation
corr = main_dataset.corr()
c1 = corr.abs().unstack()
c1.sort_values(ascending = False)[12:24:2]

#target attribute value counts
c = main_dataset.Potability.value_counts()
labels=[0,1]
print(c)

#count-target attribute- countplot
count = sns.countplot(x = "Potability", data= main_dataset)
plt.xticks(ticks = [0,1], labels =["Not Potable", "Potable"])
plt.show()

#pie chart of target attribute
fig =  px.pie (main_dataset, names = "Potability", hole = 0.4, template = "plotly")
fig.show ()

#boxplot
sns.set_theme(style="whitegrid")

sns.boxplot(y= "ph", data = main_dataset)

sns.boxplot(y= "Hardness", data = main_dataset)

sns.boxplot(y= "Solids", data = main_dataset)

sns.boxplot(y= "Chloramines", data = main_dataset)

sns.boxplot(y= "Sulfate", data = main_dataset)

sns.boxplot(y= "Conductivity", data = main_dataset)

sns.boxplot(y= "Organic_carbon", data = main_dataset)

sns.boxplot(y= "Trihalomethanes", data = main_dataset)

sns.boxplot(y= "Turbidity", data = main_dataset)


#overall value overviwe or count
main_dataset.hist(figsize=[20,10])
sns.pairplot(main_dataset, kind = "reg", diag_kind = "hist", hue = "Potability")
sns.distplot(main_dataset['Potability']) 

#visualization on respective to Potability
plt.figure(figsize=(10,10))
for ax,col in enumerate(main_dataset.columns[:9]):
    plt.subplot(3,3,ax+1)
    plt.title(f'Distribution of {col}')
    sns.kdeplot(x=main_dataset[col], fill=True, alpha=0.4, hue = main_dataset.Potability, multiple='stack')
plt.tight_layout()

main_dataset.hist(column = "ph", by = "Potability")
main_dataset.hist(column = "Hardness", by = "Potability")

fig = px.histogram(main_dataset, x = "Sulfate", facet_row = "Potability", template= 'plotly_dark')
fig.show()

fig = px.histogram(main_dataset, x = "Trihalomethanes", facet_row = "Potability", template= 'plotly_dark')
fig.show()

fig = px.histogram(main_dataset, x = "Organic_carbon", facet_row = "Potability", template= 'plotly_dark')
fig.show()

fig = px.scatter (main_dataset, x = "ph", y = "Sulfate", color=("Potability"), template = "plotly_dark",  trendline="lowess")
fig.show ()

fig = px.scatter (main_dataset, x = "ph", y = "Chloramines", color=("Potability"), template = "plotly_dark",  trendline="lowess")
fig.show ()


#mean and standard by potability
mean_potability = main_dataset.groupby("Potability").mean()
std_potability = main_dataset.groupby("Potability").std()

#data filling
main_dataset.nunique()

#fillna using mean regarding potability
main_dataset_copy['ph'] = main_dataset_copy['ph'].fillna(main_dataset_copy.groupby('Potability')['ph'].transform('mean'))
main_dataset_copy['Sulfate'] = main_dataset_copy['Sulfate'].fillna(main_dataset_copy.groupby('Potability')['Sulfate'].transform('mean'))
main_dataset_copy['Trihalomethanes'] = main_dataset_copy['Trihalomethanes'].fillna(main_dataset_copy.groupby('Potability')['Trihalomethanes'].transform('mean'))

#checking after filling
main_dataset.isnull().sum()
main_dataset_copy.isnull().sum()


#data partioning
x = main_dataset_copy.drop("Potability", axis = 1)
y = main_dataset_copy["Potability"] 

#spilitting the dataset and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=101, 
                                                    stratify= y ) 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#model building
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

#random forest
param_grid = {'n_estimators': [100, 200, 300], 'max_features': ['auto', 'sqrt'], 'bootstrap': [True, False], 'criterion':['entropy', 'gini']}
rfcgrid = GridSearchCV(rf(random_state=101), param_grid, verbose=100, cv=10, n_jobs=-2)
rfcgrid.fit(x_train, y_train)

# Best params of Random Forest
rfcgrid.best_params_

#report
rfcpredictions = rfcgrid.predict(x_test)

print("Confusion Matrix - Random Forest Using Entropy Index")
print(confusion_matrix(y_test,rfcpredictions))
mat = confusion_matrix(y_test, rfcpredictions)
axes = sns.heatmap(mat, square  = True, annot = True, fmt = 'd', cbar = True, cmap = plt.cm.RdPu)

cm3 = confusion_matrix(y_test, rfcpredictions)
sns.heatmap(cm3/np.sum(cm3), annot = True, fmt=  '0.2%', cmap = 'Reds')

print("\n")
print("Accuracy Score - Random Forest")
print(accuracy_score(y_test, rfcpredictions))

print("\n")
print("Classification Report - Random Forest")
classification_report_dataset_rf = classification_report(y_test,rfcpredictions)
print(classification_report(y_test,rfcpredictions))

print("\n")
print("F1 Score - Random Forest")
print(f1_score(y_test, rfcpredictions))


