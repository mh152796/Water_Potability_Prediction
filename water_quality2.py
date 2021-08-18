#trying another way




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


main_dataset_copy2.Potability.value_counts()

main_dataset_copy2.isnull().sum()

main_dataset_copy2['ph'] = main_dataset_copy['ph'].fillna(main_dataset_copy.groupby('Potability')['ph'].transform('mean'))
main_dataset_copy2['Sulfate'] = main_dataset_copy['Sulfate'].fillna(main_dataset_copy.groupby('Potability')['Sulfate'].transform('mean'))
main_dataset_copy2['Trihalomethanes'] = main_dataset_copy['Trihalomethanes'].fillna(main_dataset_copy.groupby('Potability')['Trihalomethanes'].transform('mean'))

main_dataset_copy2.isnull().sum()
main_dataset_copy2.Potability.value_counts()

notpotable  = main_dataset_copy2[main_dataset_copy2['Potability']==0]
potable = main_dataset_copy2[main_dataset_copy2['Potability']==1]  

from sklearn.utils import resample
df_minority_upsampled = resample(potable, replace = True, n_samples = 1988) 



from sklearn.utils import shuffle
main_dataset_copy2 = pd.concat([notpotable, df_minority_upsampled])
main_dataset_copy2 = shuffle(main_dataset_copy2) 

main_dataset_copy2.Potability.value_counts()

X = main_dataset_copy2.drop("Potability", axis = 1)
Y = main_dataset_copy2["Potability"] 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state=101, 
                                                    stratify= Y ) 


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

params_rf = {'n_estimators':[100,200, 350, 500], 'min_samples_leaf':[2, 10, 30]}
grid_rf = GridSearchCV(rf(random_state = 101), param_grid=params_rf, cv=5)

grid_rf.fit(X_train, Y_train)

grid_rf.best_params_

rfcpredictions = grid_rf.predict(X_test)

print("Confusion Matrix - Random Forest Using Entropy Index")
print(confusion_matrix(Y_test,rfcpredictions))
mat = confusion_matrix(Y_test, rfcpredictions)
axes = sns.heatmap(mat, square  = True, annot = True, fmt = 'd', cbar = True, cmap = plt.cm.RdPu)

cm3 = confusion_matrix(Y_test, rfcpredictions)
sns.heatmap(cm3/np.sum(cm3), annot = True, fmt=  '0.2%', cmap = 'Reds')

print("\n")
print("Accuracy Score - Random Forest")
print(accuracy_score(Y_test, rfcpredictions))

print("\n")
print("Classification Report - Random Forest")
classification_report_dataset_rf = classification_report(Y_test,rfcpredictions)
print(classification_report(Y_test,rfcpredictions))

print("\n")
print("F1 Score - Random Forest")
print(f1_score(Y_test, rfcpredictions))
