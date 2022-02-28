# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 15:28:59 2022
@author: camillo.majerczyk
""" 
#%% Resources
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

#%% Data Preparation
data = pd.read_csv(r"C:\Users\camillo.majerczyk\Desktop\STAT_LEARN_Project\Health_csv_Data_Majerczyk.csv", sep = ';')
data['Wheelchair'] = data['Wheelchair'].fillna(0)
data.shape
data['Wheelchair'] = data['Wheelchair'].astype('int64')
data.dtypes

#%%
data.hist(column="Hospi_time", bins=25, grid=False, figsize=(12,8), color='#ef7d00', zorder=2, rwidth=0.9)
short_hospi_time = data[data["Hospi_time"]<365]
short_hospi_time.hist(column="Hospi_time", bins=25, grid=False, figsize=(12,8), color='#ef7d00', zorder=2, rwidth=0.9)
medium_hospi_time = data[data["Hospi_time"].between(365, 3285)]
medium_hospi_time.hist(column="Hospi_time", bins=25, grid=False, figsize=(12,8), color='#ef7d00', zorder=2, rwidth=0.9)

#%% assign labels to Hospi_time
def assign_label(i):
    if i >= 0 and i <= 14:
        return 1
    if i > 14 and i <= 90:
        return 2
    if i > 90 and i <= 365:
        return 3
    if i > 365 and i <= 720:
        return 4
    if i > 720 and i <= 1440:
        return 5
    if i > 1440 and i <= 2160:
        return 6
    if i > 2160 and i <= 3240:
        return 7
    if i > 3240:
        return 8

data["Hospi_time"] = data["Hospi_time"].apply(assign_label)

#%% Labels distribution
barWidth = 0.9
bar1 = len(data[(data["Hospi_time"] == 1)])
bar2 = len(data[(data["Hospi_time"] == 2)])
bar3 = len(data[(data["Hospi_time"] == 3)])
bar4 = len(data[(data["Hospi_time"] == 4)])
bar5 = len(data[(data["Hospi_time"] == 5)])
bar6 = len(data[(data["Hospi_time"] == 6)])
bar7 = len(data[(data["Hospi_time"] == 7)])
bar8 = len(data[(data["Hospi_time"] == 8)])
#%% barplot
plt.bar(x=1, height= bar1, width = barWidth, color = '#ef7d00', label='0_14_days')
plt.bar(x=2, height= bar2, width = barWidth, color = '#00b2c4', label='15_90_days')
plt.bar(x=3, height= bar3, width = barWidth, color = '#ffd203', label='91_365_days')
plt.bar(x=4, height= bar4, width = barWidth, color = '#92bd1f', label='366_720_days')
plt.bar(x=5, height= bar5, width = barWidth, color = '#92bd1f', label='721_1440_days')
plt.bar(x=6, height= bar6, width = barWidth, color = '#ffd203', label='1441_2160_days')
plt.bar(x=7, height= bar7, width = barWidth, color = '#00b2c4', label='2161_3240_days')
plt.bar(x=8, height= bar8, width = barWidth, color = '#ef7d00', label='More_than_3240')
plt.xticks([r + barWidth for r in range(8)], ['0_14_days', '15_90_days', '91_365_days', '366_720_days', '721_1440_days', '1441_2160_days', '2161_3240_days', 'More_than_3240'], rotation=45, size=8)

#%% Correlation
corr = data.corr(method='spearman')
corr
#%% Plot corr heatmap
heatmap_col = seaborn.diverging_palette(186, 49, l=65, center="light", as_cmap=True)
seaborn.set(font_scale=0.5)
mask = numpy.zeros_like(corr)
mask[numpy.triu_indices_from(mask)] = True
seaborn.heatmap(corr, mask=mask, square=True, cmap= heatmap_col, linewidths=1, linecolor='white')

#%% Random Forest var ranking
dataTree = data.drop('Id_Hospitalization', 1)
train_features, test_features, train_labels, test_labels = train_test_split(dataTree, dataTree["Hospi_time"],test_size=0.2)
Treemodel = RandomForestClassifier(n_estimators = 100)
Treemodel.fit(train_features, train_labels)
importances = Treemodel.feature_importances_[1:]
indices = numpy.argsort(Treemodel.feature_importances_[1:])
features = dataTree.columns
#%% plot features ranking
ax=plt.axes()
ax.set_facecolor('#f0f0f0')
plt.barh(range(len(indices)), importances[indices], color='#ef7d00')
plt.yticks(range(len(indices)), features[indices])

#%% TREE CLASSIFIER

rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500, 800],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,8,16,24,30],
    'criterion' :['gini', 'entropy']
}
#Hyperparameters tuning
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, return_train_score=True)
CV_rfc.fit(train_features, train_labels)
CV_rfc.best_params_

rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=24, criterion='gini')
rfc1.fit(train_features, train_labels)
pred=rfc1.predict(test_features)

#rfc1 accuracy score on testset
print("Accuracy of rfc1: ",accuracy_score(test_labels,pred))

#%% Plot GridSearchCV results
def plot_search_results(grid):
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']
    ## Getting indexes of values per hyperparameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))
    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = numpy.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = numpy.where(best_parms_mask)[0]
        x = numpy.array(params[p])
        y_1 = numpy.array(means_test[best_index])
        e_1 = numpy.array(stds_test[best_index])
        y_2 = numpy.array(means_train[best_index])
        e_2 = numpy.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()

plot_search_results(CV_rfc)

#%% Evaluation of the model
#Accuracy evaluation on training set
nested_score_trainset = cross_val_score(rfc1, X=train_features, y=train_labels, cv=5)
print('rfc1 Nested CV accuracy on training set: %.3f +/- %.3f' % (numpy.mean(nested_score_trainset), numpy.std(nested_score_trainset)))
#Accuracy evalaution on test set
nested_score_testset = cross_val_score(rfc1, X=test_features, y=test_labels, cv=5)
print('rfc1 Nested CV accuracy on test set: %.3f +/- %.3f' % (numpy.mean(nested_score_testset), numpy.std(nested_score_testset)))

#%% Plot confusion matrix
#On testset
matrix = sklearn.metrics.confusion_matrix(test_labels, pred)
seaborn.set(font_scale=0.6)
seaborn.heatmap(matrix, annot=True, annot_kws={'size':8}, cmap=plt.cm.Blues)
class_names = ['0_14_days', '15_90_days', '91_365_days', '366_720_days', '721_1440_days', '1441_2160_days', '2161_3240_days', 'More_than_3240']
tick_marks = numpy.arange(len(class_names))
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

#On trainset
pred_train=rfc1.predict(train_features)
print("Accuracy for rfc1 on training set: ",accuracy_score(train_labels, pred_train))
matrix2 = sklearn.metrics.confusion_matrix(train_labels, pred_train)
seaborn.set(font_scale=0.6)
seaborn.heatmap(matrix2, annot=True, annot_kws={'size':8}, cmap=plt.cm.Oranges)
class_names = ['0_14_days', '15_90_days', '91_365_days', '366_720_days', '721_1440_days', '1441_2160_days', '2161_3240_days', 'More_than_3240']
tick_marks = numpy.arange(len(class_names))
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

#%% Validation curve for max depth
train_scores_vc, test_scores_vc = validation_curve(rfc1, train_features, train_labels, param_name="max_depth", param_range= range(4,28,4), cv=5)
train_mean_vc = numpy.mean(train_scores_vc, axis=1)
train_std_vc = numpy.std(train_scores_vc, axis=1)
test_mean_vc = numpy.mean(test_scores_vc, axis=1)
test_std_vc = numpy.std(test_scores_vc, axis=1)
param_range = range(4, 28, 4)
#plot val curve
plt.plot(param_range, train_mean_vc, label="Training score", color="orange")
plt.plot(param_range, test_mean_vc, label="Test score", color="blue")
plt.fill_between(param_range, train_mean_vc - train_std_vc, train_mean_vc + train_std_vc, color="gray")
plt.fill_between(param_range, test_mean_vc - test_std_vc, test_mean_vc + test_std_vc, color="gainsboro")
plt.title("Validation Curve With Random Forest")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

#%% rfc2 classifier
#Change max_depth trying to solve overfitting
rfc2=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=12, criterion='gini')
rfc2.fit(train_features, train_labels)
pred2=rfc2.predict(test_features)
print("rfc2 ccuracy for Random Forest on test set: ",accuracy_score(test_labels,pred2))

matrix3 = sklearn.metrics.confusion_matrix(test_labels, pred2)
seaborn.set(font_scale=0.6)
seaborn.heatmap(matrix3, annot=True, annot_kws={'size':8}, cmap=plt.cm.Blues)
class_names = ['0_14_days', '15_90_days', '91_365_days', '366_720_days', '721_1440_days', '1441_2160_days', '2161_3240_days', 'More_than_3240']
tick_marks = numpy.arange(len(class_names))
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('rfc2 Confusion Matrix on test set')
plt.show()

#Confusion matrix on training set
pred4= rfc2.predict(train_features)
matrix4= sklearn.metrics.confusion_matrix(train_labels, pred4)
print("rfc2 ccuracy on training set: ",accuracy_score(train_labels,pred4))

seaborn.set(font_scale=0.6)
seaborn.heatmap(matrix4, annot=True, annot_kws={'size':8}, cmap=plt.cm.Oranges)
class_names = ['0_14_days', '15_90_days', '91_365_days', '366_720_days', '721_1440_days', '1441_2160_days', '2161_3240_days', 'More_than_3240']
tick_marks = numpy.arange(len(class_names))
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('rfc2 Confusion Matrix on training set')
plt.show()

#Evaluation rfc2
nested_score_testset_rfc2 = cross_val_score(rfc2, X=test_features, y=test_labels, cv=5)
print('rfc2 Nested CV accuracy on test set: %.3f +/- %.3f' % (numpy.mean(nested_score_testset_rfc2), numpy.std(nested_score_testset_rfc2)))

#%% NEURAL NETWORKS
#Setting up the data
data_NN = dataTree
data_NN = data_NN.astype('float')
data_NN.dtypes
train_features_NN, test_features_NN, train_labels_NN, test_labels_NN = train_test_split(dataTree, dataTree["Hospi_time"],test_size=0.2)
train_features_NN = train_features_NN.astype('float')
train_labels_NN = train_labels_NN.astype('float')
test_features_NN = test_features_NN.astype('float')
test_labels_NN = test_labels_NN.astype('float')

#%% Define NN model with SGD activation function
def get_mlp_model(hiddenLayerOne=784, hiddenLayerTwo=256, dropout=0.2, learnRate=0.01):
    model = Sequential() #initialize a sequential model
    model.add(Flatten()) #initialize layer to flatten the input data
    model.add(Dense(hiddenLayerOne, activation="relu", input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(hiddenLayerTwo, activation="relu"))
    model.add(Dropout(dropout))
    # add a softmax layer on top
    model.add(Dense(10, activation="softmax"))
    # compile the model
    model.compile(optimizer=SGD(learning_rate=learnRate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # return compiled model
    return model

#%% Baseline NN model with SGD
#initialize the model with default hyperparameters values
model = get_mlp_model()
# train the network (no hyperparameter tuning)
H = model.fit(x=train_features_NN.values, y=train_labels_NN.values, validation_data=(test_features_NN, test_labels_NN), batch_size=8, epochs=20)
# Predictions on the test set
accuracy = model.evaluate(test_features_NN, test_labels_NN)
accuracy
print("Baseline NN accuracy on test set: {:.2f}%".format(accuracy[1] * 100))

#%% NN Hyperparameters tuning
# wrap our model into a scikit-learn compatible classifier
model = KerasClassifier(build_fn=get_mlp_model, verbose=0)
#grid of hyperparameters search space
hiddenLayerOne = [256, 512, 784]
hiddenLayerTwo = [128, 256, 512]
learnRate = [1e-2, 1e-3, 1e-4]
dropout = [0.3, 0.4, 0.5]
batchSize = [4, 8, 16, 32]
epochs = [10, 20, 30, 40]
#Dictionary from the grid
grid = dict(
	hiddenLayerOne=hiddenLayerOne,
	learnRate=learnRate,
	hiddenLayerTwo=hiddenLayerTwo,
	dropout=dropout,
	batch_size=batchSize,
	epochs=epochs
)

#Random search with 3-fold CV and  hyperparameter tuning
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3, param_distributions=grid, scoring="accuracy")
searchResults = searcher.fit(train_features_NN.values, train_labels_NN.values)
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("The best score is {:.2f} using {}".format(bestScore, bestParams))

#%% Evaluate the best NN with SGD
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(test_features_NN, test_labels_NN)
print("accuracy: {:.2f}%".format(accuracy * 100))

#%% Define NN model with Adam activation function
def get_mlp_model2(hiddenLayerOne=784, hiddenLayerTwo=256, dropout=0.2, learnRate=0.01):
    model = Sequential() #initialize a sequential model
    model.add(Flatten()) #initialize layer to flatten the input data
    model.add(Dense(hiddenLayerOne, activation="relu", input_shape=(784,)))
    model.add(Dropout(dropout))
    model.add(Dense(hiddenLayerTwo, activation="relu"))
    model.add(Dropout(dropout))
    # add a softmax layer on top
    model.add(Dense(10, activation="softmax"))
    # compile the model
    model.compile(optimizer=Adam(learning_rate=learnRate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # return compiled model
    return model

#%% Baseline NN model with Adam
model2 = get_mlp_model2()
K = model2.fit(x=train_features_NN.values, y=train_labels_NN.values, validation_data=(test_features_NN, test_labels_NN), batch_size=8, epochs=20)
#evaluate on test set
accuracy2 = model2.evaluate(test_features_NN, test_labels_NN)
accuracy2
print("Second Baseline NN accuracy: {:.2f}%".format(accuracy2[1] * 100))

#%% NN Hyperparameters tuning
model3 = KerasClassifier(build_fn=get_mlp_model2, verbose=0)
#Random search with a 3-fold CV and hyperparameter tuning
searcher3 = RandomizedSearchCV(estimator=model3, n_jobs=-1, cv=3, param_distributions=grid, scoring="accuracy")
searchResults3 = searcher3.fit(train_features_NN.values, train_labels_NN.values)
bestScore3 = searchResults3.best_score_
bestParams3 = searchResults3.best_params_
print("Best score is {:.2f} using {}".format(bestScore3, bestParams3))

#%%
#Evaluation best model with Adam
bestModel3 = searchResults3.best_estimator_
accuracy3 = bestModel3.score(test_features_NN, test_labels_NN)
print("accuracy: {:.2f}%".format(accuracy3 * 100))

#%% Confusion matrix for results of NN with Adam
pred_NN_Adam= bestModel3.predict(test_features)
matrix_NN_Adam= sklearn.metrics.confusion_matrix(test_labels, pred_NN_Adam)
matrix_NN_Adam
# plot 
seaborn.set(font_scale=0.6)
seaborn.heatmap(matrix_NN_Adam, annot=True, annot_kws={'size':8}, cmap=plt.cm.Blues)
class_names = ['0_14_days', '15_90_days', '91_365_days', '366_720_days', '721_1440_days', '1441_2160_days', '2161_3240_days', 'More_than_3240']
tick_marks = numpy.arange(len(class_names))
tick_marks2 = tick_marks
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for NN Adam')
plt.show()













