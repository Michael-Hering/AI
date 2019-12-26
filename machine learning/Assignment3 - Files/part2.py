# import pandas as pd
# import numpy as np
# import pickle
# import sklearn.preprocessing as preprocessing

# le = preprocessing.LabelEncoder()

# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score

# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GridSearchCV

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix

# skf = StratifiedKFold(n_splits=10)

# Label = "Credit"
# Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

# def saveBestModel(clf):
#     pickle.dump(clf, open("bestModel.model", 'wb'))

# def readData(file):
#     df = pd.read_csv(file)
#     return df

# def trainOnAllData(df, clf):
#     #Use this function for part 4, once you have selected the best model
#     print("TODO")

#     saveBestModel(clf)

# df = readData("credit_train.csv")
# y = df.iloc[:,-1]
# X = df.iloc[:,0:-1]

# classifiers = {"Logistic Regression": LogisticRegression(),
#                 "Naive Bayes": GaussianNB(),
#                 "SVM": SVC(probability=True), # probability is neccesary for AUROC
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "Random Forest": RandomForestClassifier(),
#                 "KNN": KNeighborsClassifier(),
#                 "ADA Boost": AdaBoostClassifier()
#                }


# ############################
# ### BASIC MODEL BUILDING ###
# ############################

# classifier_scores = {}
# for name, clf in zip(classifiers.keys(), classifiers.values()):
#   print("Performing K-Fold validation on ", name)
#   classifier_scores[name] = {}
#   classifier_scores[name]["AUROC_scores"] = []
#   classifier_scores[name]["clf"] = clf
  
#   accuracy_scores = []
#   precision_scores = []
#   recall_scores = []
#   i = 0
#   for train_indicies, valid_indicies in skf.split(X, y): 
#     X_train = X.iloc[train_indicies]
#     y_train = y[train_indicies]
#     X_valid = X.iloc[valid_indicies,:]
#     y_valid = y[valid_indicies]

#     fitted = clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_valid)
#     probs = clf.predict_proba(X_valid)

#     probs = probs[:, 1]
#     AUROC = roc_auc_score(y_valid, probs)
#     accuracy = accuracy_score(y_valid, y_pred)
#     recall = recall_score(y_valid, y_pred, pos_label="good")
#     precision = precision_score(y_valid, y_pred, pos_label="good")

#     classifier_scores[name]["AUROC_scores"].append(AUROC)
#     accuracy_scores.append(accuracy)
#     precision_scores.append(recall)
#     recall_scores.append(precision)

#   classifier_scores[name]["accuracy_score"] = np.mean(accuracy_scores)
#   classifier_scores[name]["precision_score"] = np.mean(precision_scores)
#   classifier_scores[name]["recall_score"] = np.mean(recall_scores)
#   classifier_scores[name]["AUROC_mean"] = np.mean(classifier_scores[name]["AUROC_scores"])
#   classifier_scores[name]["AUROC_std"] = np.std(classifier_scores[name]["AUROC_scores"])

# #############################
# ### HYPERPARAMETER TUNING ###
# #############################

# svc_param_grid = [
#   {"C": [0.0001, 0.001, .01, 0.02], "gamma": [0.0001, 0.001, .01, 0.02]}
# ]
# svc_grid_search = GridSearchCV(classifiers["SVM"], svc_param_grid, cv = 5)
# svc_grid_search.fit(X, y)
# svc_params = svc_grid_search.best_params_

# forest_param_grid = [
#   {"max_depth": [2, 4, 6, 8, 10], "n_estimators": [8, 32, 64, 128]}
# ]
# forest_grid_search = GridSearchCV(classifiers["Random Forest"], forest_param_grid, cv = 5)
# forest_grid_search.fit(X, y)
# forest_params = forest_grid_search.best_params_

# print("Best parameters for SVM: ", svc_params)
# print("Best parameters for Random Forest: ", forest_params)

# tuned_classifiers = { "Tuned SVC": SVC(probability=True, **svc_params), # probability is neccesary for AUROC
#                 "Tuned Random Forest": RandomForestClassifier(**forest_params),
#                }

# for name, clf in zip(tuned_classifiers.keys(), tuned_classifiers.values()):
#   print("Performing K-Fold validation on ", name)
#   classifier_scores[name] = {}
#   classifier_scores[name]["AUROC_scores"] = []
#   classifier_scores[name]["clf"] = clf
  
#   accuracy_scores = []
#   precision_scores = []
#   recall_scores = []
#   i = 0
#   for train_indicies, valid_indicies in skf.split(X, y): 
#     X_train = X.iloc[train_indicies]
#     y_train = y[train_indicies]
#     X_valid = X.iloc[valid_indicies,:]
#     y_valid = y[valid_indicies]

#     fitted = clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_valid)
#     probs = clf.predict_proba(X_valid)

#     probs = probs[:, 1]
#     AUROC = roc_auc_score(y_valid, probs)
#     accuracy = accuracy_score(y_valid, y_pred)
#     recall = recall_score(y_valid, y_pred, pos_label="good")
#     precision = precision_score(y_valid, y_pred, pos_label="good")

#     classifier_scores[name]["AUROC_scores"].append(AUROC)
#     accuracy_scores.append(accuracy)
#     precision_scores.append(recall)
#     recall_scores.append(precision)

#   classifier_scores[name]["accuracy_score"] = np.mean(accuracy_scores)
#   classifier_scores[name]["precision_score"] = np.mean(precision_scores)
#   classifier_scores[name]["recall_score"] = np.mean(recall_scores)
#   classifier_scores[name]["AUROC_mean"] = np.mean(classifier_scores[name]["AUROC_scores"])
#   classifier_scores[name]["AUROC_std"] = np.std(classifier_scores[name]["AUROC_scores"])

# ##############################
# ### ANALYSIS OF BEST MODEL ###
# ##############################

# # Use AUROC as performance metric
# best_classifier = ""
# max_AUROC = -1
# for name, data in zip(classifier_scores.keys(), classifier_scores.values()):
#   if data["AUROC_mean"] > max_AUROC:
#     max_AUROC = data["AUROC_mean"]
#     best_classifier = name

# print("Best classifier: ", best_classifier)
# print("Accuracy: ", classifier_scores[best_classifier]["accuracy_score"])
# print("Precision: ", classifier_scores[best_classifier]["precision_score"])
# print("Recall: ", classifier_scores[best_classifier]["recall_score"])

# best_clf = classifier_scores[best_classifier]["clf"].fit(X, y)
# y_pred = best_clf.predict(X)
# cMatrix = confusion_matrix(y, y_pred)
# print("Confusinon matrix: ", cMatrix)

# output = df.copy()
# output["Predicted"] = y_pred
# print(output)
# output.to_csv(r'bestModel.output', index=False)

# print(best_clf)
# saveBestModel(best_clf)