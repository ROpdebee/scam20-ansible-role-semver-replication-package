import pickle
import pydotplus
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    dataframe = pd.read_csv("data/metrics_diffs_releases_merged.csv")

    # Checking rows with null values
    data_null_values = dataframe[dataframe.isna().any(axis=1)]
    len_null = len(data_null_values)
    len_data = len(dataframe)

    # Print message
    print(f"{len_null} out of {len_data} contains at least one null value", end=" ")
    print(f"representing {len_null / len_data} % of the instances")

    # Cleaning data
    cleaned_data = dataframe.dropna(how='any', axis=0)
    float_columns = cleaned_data.select_dtypes(include=['float64'])

    for col in float_columns.columns.values:
        cleaned_data[col] = cleaned_data[col].astype('int64')

    # Discarding the first three columns and the column with all values as zero
    filtered_data = cleaned_data.drop(['id', 'v1', 'v2', 'HandlersFileRelocation'], axis=1)

    # Correlation analysis
    correlation_matrix = filtered_data.corr()
    unstacked_correlation_matrix = correlation_matrix.unstack()
    sorted_values = unstacked_correlation_matrix.sort_values(kind="quicksort")
    print(sorted_values[1550:1560])

    # Droping second correlated feature
    filtered_data = filtered_data.drop(['HandlersFileAddition'], axis=1)

    # Transforming into categorical the classes to be predicted
    to_transform = {"release": {"patch": 0, "minor": 1, "major": 2}}
    filtered_data.replace(to_transform, inplace=True)

    # Transforming data into Numpy arrays
    X = filtered_data[filtered_data.columns[:-1]].to_numpy()
    y = filtered_data[filtered_data.columns[-1]].to_numpy()

    # Print shapes
    print("Shapes of the data:")
    print(X.shape)
    print(y.shape)

    # Configuring classifier
    clf = RandomForestClassifier(n_jobs=8, max_leaf_nodes=200)

    # Plot relations between the number of features and the accuracy of the model
    rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    column_names = filtered_data.columns[:-1].tolist()
    selected_features = list()

    for i, value in enumerate(rfecv.support_):
        if value:
            selected_features.append(column_names[i])

    print("Original Features on Data: ", column_names)
    print("Ranking of Features: ", rfecv.ranking_)
    print("Selected Features: ", selected_features)
    print("Discarded Features: ", set(column_names).difference(set(selected_features)))

    X = filtered_data[selected_features].to_numpy()
    # y remains the same

    # Stratified Training
    cv = StratifiedKFold(n_splits=10)

    precision_scores = list()
    recall_scores = list()
    accuracy_scores = list()
    confusion_matrices = list()
    feature_importances = np.zeros(39)

    for i, (train, test) in enumerate(cv.split(X, y)):
        print(f"Fold # {i + 1}")

        print("Training ...")
        clf.fit(X[train], y[train])
        print("Done!")

        y_predict = clf.predict(X[test])

        precision_value = precision_score(y[test], y_predict, average="macro")
        recall_value = recall_score(y[test], y_predict, average="macro")
        accuracy_value = accuracy_score(y[test], y_predict)
        conf_matrix = confusion_matrix(y[test], y_predict)
        feature_importances += clf.feature_importances_

        print(f"Precision: {precision_value}\nRecall: {recall_value}\n Accuracy: {accuracy_value}")
        print(conf_matrix)

        precision_scores.append(precision_value)
        recall_scores.append(recall_value)
        accuracy_scores.append(accuracy_value)
        confusion_matrices.append(conf_matrix)

    print("Mean of Precisions: ", np.mean(precision_scores))
    print("Mean of Recalls: ", np.mean(recall_scores))
    print("Mean of Accuracies: ", np.mean(accuracy_scores))

    print("Confusion matrix:")
    print(sum(confusion_matrices))

    # print("Weights of the features:")
    # print(feature_importances / 10)

    clf.fit(X, y)
    pickle.dump(clf, open("data/randomForest.pickle", "wb"))

    fn = selected_features
    cn = ["patch", "minor", "major"]

    # Saving all trees in the forest
    for i, estimator in enumerate(clf.estimators_):
        dot_data = tree.export_graphviz(estimator,
                                        out_file=None,
                                        feature_names = fn,
                                        class_names=cn,
                                        filled=True,
                                        rounded=True)

        graph2 = pydotplus.graph_from_dot_data(dot_data)  
        graph2.write_pdf(f"images/trees/tree_{i}.pdf")
