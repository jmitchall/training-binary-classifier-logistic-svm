import os
import time

import numpy as np
import pandas as pd
from sklearn import multiclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

http_proxy = 'http://http.proxy.fmr.com:8000/'
https_proxy = 'http://http.proxy.fmr.com:8000/'
no_proxy = '169.254.169.254'
os.environ['http_proxy'] = http_proxy
os.environ['https_proxy'] = https_proxy
os.environ['no_proxy'] = no_proxy
os.environ['HTTP_PROXY'] = http_proxy
os.environ['HTTPS_PROXY'] = https_proxy


def get_memory_usage_of_data_frame(df, bytes_to_mb_div=0.000001):
    mem = round(df.memory_usage().sum() * bytes_to_mb_div, 3)
    return_str = "Memory usage is " + str(mem) + " MB"

    return return_str


def convert_to_sparse_pandas(df, exclude_columns):
    """
    https://towardsdatascience.com/working-with-sparse-data-sets-in-pandas-and-sklearn-d26c1cfbe067
    Converts columns of a data frame into SparseArrays and returns the data frame with transformed columns.
    Use exclude_columns to specify columns to be excluded from transformation.
    :param df: pandas data frame
    :param exclude_columns: list
        Columns not be converted to sparse
    :return: pandas data frame
    """
    from pandas.arrays import SparseArray
    pd.DataFrame.iteritems = pd.DataFrame.items
    df = df.copy()
    exclude_columns = set(exclude_columns)
    # get iterable tuple of column name and column data from data frame
    for (columnName, columnData) in df.iteritems():
        if columnName in exclude_columns:
            continue
        df[columnName] = SparseArray(columnData.values, dtype='uint8')
    return df


def data_frame_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    from scipy.sparse import lil_matrix
    import numpy as np

    # Initialize a sparse matrix with the same shape as the DataFrame `df`.
    # The `lil_matrix` is a type of sparse matrix provided by SciPy.
    # It's good for incremental construction. Row-based LIst of Lists sparse matrix.
    arr = lil_matrix(df.shape, dtype=np.float32)

    # Iterate over each column in the DataFrame.
    for i, col in enumerate(df.columns):
        # Create a boolean mask where each element is `True` if the corresponding element in the column is not zero,
        # and `False` otherwise.
        ix = df[col] != 0
        # Set the value of the sparse matrix at the positions where the mask is `True` to 1.
        # The `np.where(ix)` function returns the indices where `ix` is `True`.
        arr[np.where(ix), i] = 1

    # Convert the `lil_matrix` to a `csr_matrix` (Compressed Sparse Row matrix) and return it.
    # The `csr_matrix` is another type of sparse matrix that is efficient for arithmetic operations
    # and is suitable for machine learning algorithms in SciPy and sklearn.
    return arr.tocsr()


def get_csr_memory_usage(x_csr, bytes_to_mb_div=0.000001):
    mem = (x_csr.data.nbytes + x_csr.indptr.nbytes + x_csr.indices.nbytes) * bytes_to_mb_div
    return "Memory usage is " + str(mem) + " MB"


def select_k_best_features(df_full_data, top_k=10):
    """
    Selects the K best features of the data frame using chi2
    1. Univariate Selection
       Statistical tests can be used to select those features that have the strongest relationship with the output variable.
    :param df_full_data: must be a pandas data frame that has the target column as the last column
    :param k:  number of best features to select
    :return:
    """
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X = df_full_data.iloc[:, 0:df_full_data.shape[1] - 1]
    Y = df_full_data.iloc[:, df_full_data.shape[1] - 1]
    bestfeatures = SelectKBest(score_func=chi2, k=top_k)
    fit = bestfeatures.fit(X, Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    return featureScores.nlargest(top_k, 'Score')


def select_feature_importance(df_full_data, top_k=10):
    """
    Selects the K best features of the data frame using chi2
    2. Feature Importance
       You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
    :param df_full_data: must be a pandas data frame that has the target column as the last column
    :param k:  number of best features to select
    :return:
    """
    from sklearn.ensemble import ExtraTreesClassifier
    X = df_full_data.iloc[:, 0:df_full_data.shape[1] - 1]
    Y = df_full_data.iloc[:, df_full_data.shape[1] - 1]
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(top_k).plot(kind='barh')
    # return top_k features
    return feat_importances.nlargest(top_k)


def select_correlation_features(df_full_data):
    """
    Selects the K best features of the data frame using chi2
    3. Correlation Matrix with Heatmap
       Correlation states how the features are related to each other or the target variable.
    :param df_full_data: must be a pandas data frame that has the target column as the last column
    :param k:  number of best features to select
    :return:
    """
    # https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
    # Using Pearson Correlation
    import seaborn as sns
    import matplotlib.pyplot as plt
    corrmat = df_full_data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(df_full_data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    # select top k features that most independent of each other
    g.get_figure().savefig('correlation_heatmap.png')


def get_class_weights(df_classes):
    """
    Implementing class weights
    If the positive and negative cases in the dataset are imbalanced
    (e.g., there are significantly more negative cases than positive cases),
    then the model may be biased towards the more prevalent class.
    Implementing class weights (i.e., giving more weight to the minority class) can
    help balance the precision and recall of the model.
    :param df_classes: must be a pandas data frame that has the target column as the last column
    :return:
    """
    from sklearn.utils import class_weight
    Y = df_classes.iloc[:, df_classes.shape[1] - 1]
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    #convert Class weights to dictionary
    class_weights = dict(enumerate(class_weights))
    print("Class weights: ", class_weights)
    return class_weights


# python main entry
if __name__ == '__main__':
    adult_data = pd.read_csv('adult.csv', sep=",")
    Y_Data = pd.read_csv('adult_Y_data_encode_binary.csv', sep=",")
    all_Data_df = pd.read_csv('adult_all_data_processed.csv', sep=",")
    all_Data_mean_encoded_df = pd.read_csv('adult_all_data_target_encoded.csv', sep=",")
    X_Data_label_encoded = pd.read_csv('adult_X_columns_label_encoded.csv', sep=",")
    X_Data_one_hot_encoded = pd.read_csv('adult_X_data_one_hot_encoded.csv', sep=",")
    X_Data_sparse_one_hot_encoded = convert_to_sparse_pandas(pd.read_csv('adult_X_sparse_one_hot_encoded.csv', sep=","),
                                                             [])
    Y_Data = pd.read_csv('adult_Y_sparse_one_hot_encoded.csv', sep=",")
    Y_Data_sparse_one_hot_encoded = convert_to_sparse_pandas(Y_Data, [])
    from scipy import sparse

    Y_Data_csr = sparse.load_npz('adult_Y_sparse_one_hot_encoded.npz')
    X_Data_csr = sparse.load_npz('adult_X_sparse_one_hot_encoded.npz')

    print('X Data one hot encoded takes up', get_memory_usage_of_data_frame(X_Data_one_hot_encoded))
    print('X sparse Data takes up', get_memory_usage_of_data_frame(X_Data_sparse_one_hot_encoded))
    print('X csr Data takes up', get_csr_memory_usage(X_Data_csr))
    print('Y Data takes up', get_memory_usage_of_data_frame(Y_Data))
    print('Y sparse Data takes up', get_memory_usage_of_data_frame(Y_Data_sparse_one_hot_encoded))
    print('Y csr Data takes up', get_csr_memory_usage(Y_Data_csr))

    # https://towardsdatascience.com/working-with-sparse-data-sets-in-pandas-and-sklearn-d26c1cfbe067
    # https://stephenleo.github.io/data-science-blog/data-science-blog/ml/feature_engineering.html#dirty-cat

    # select the best features of the data all_Data_mean_encoded_df that select target 'income'
    # https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
    # select all numeric columns

    # vector_dict = {'Pandas ONE-HOT FULL dataframe': [X_Data_one_hot_encoded, Y_Data],
    #                'Sparse pandas dataframe': [X_Data_sparse_one_hot_encoded, Y_Data_sparse_one_hot_encoded],
    #                'Scipy sparse matrix': [X_Data_csr, Y_Data_csr]
    #                }
    # numeric_columns = all_Data_mean_encoded_df.select_dtypes(
    #     include=['int64', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16']).columns
    # numeric_df = all_Data_mean_encoded_df[numeric_columns]
    # print("Numeric columns: ", numeric_columns)
    # for top_range in range(1 ,numeric_columns.size):
    #     print("Top ", top_range, " best features")
    #     spec_column , score_column = select_k_best_features(numeric_df, top_range).values.T  # transpose the values
    #     vector_dict[f'Pandas MEAN top {top_range} {spec_column}'] = [all_Data_mean_encoded_df[spec_column], Y_Data]

    # numeric_columns = X_Data_label_encoded.select_dtypes(
    #     include=['int64', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16']).columns
    # numeric_df = pd.concat( [X_Data_label_encoded[numeric_columns], Y_Data], axis=1)
    # print("Numeric columns: ", numeric_columns)
    # for top_range in range(1, numeric_columns.size):
    #     print("Top ", top_range, " best features")
    #     spec_column, score_column = select_k_best_features(numeric_df, top_range).values.T  # transpose the values
    #     vector_dict[f'Pandas Labeled top {top_range} {spec_column}'] = [X_Data_label_encoded[spec_column], Y_Data]

    vector_dict = dict()
    numeric_columns = X_Data_one_hot_encoded.select_dtypes(
        include=['int64', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16']).columns
    numeric_df = pd.concat([X_Data_one_hot_encoded[numeric_columns], Y_Data], axis=1)
    print("Numeric columns: ", numeric_columns)
    # for top_range in range(55, 58):
    #     print("Top ", top_range, " best features")
    #     spec_column, score_column = select_k_best_features(numeric_df, top_range).values.T  # transpose the values
    #     vector_dict[f'Pandas onehot top {top_range} {spec_column}'] = [X_Data_one_hot_encoded[spec_column], Y_Data]
    top_range = 56
    print("Top ", top_range, " best features")
    spec_column, score_column = select_k_best_features(numeric_df, top_range).values.T  # transpose the values
    vector_dict[f'Pandas onehot top {top_range} {spec_column}'] = [X_Data_one_hot_encoded[spec_column], Y_Data]

    important_features = select_feature_importance(numeric_df, top_range)
    # print Column names of the important features
    print("Important features: ",
          important_features.index.values)
    # does the important features contain all the spec_column values
    if not all(elem in important_features.index.values for elem in spec_column):
        # display the difference
        print("Difference: ", set(spec_column) - set(important_features.index.values))
        spec_column = list(important_features.index.values)
        vector_dict[f'Important features {len(spec_column)} {spec_column}'] = [X_Data_one_hot_encoded[spec_column],
                                                                               Y_Data]

    # check for Y_Data imbalance
    print("Y_Data value counts: ", Y_Data.value_counts())
    print("Y_Data sparse one hot encoded value counts: ", Y_Data_sparse_one_hot_encoded.value_counts())
    print("Y_Data csr value counts: ", pd.Series(Y_Data_csr.toarray().ravel()).value_counts())

    # if minority class is less than 15% of the total data, then the data is imbalanced
    minority_class = Y_Data.value_counts().min()
    imbalance_threshold = 0.2
    imbalance_threshold_data_count = imbalance_threshold * Y_Data.shape[0]
    imbalance_detected =  minority_class < imbalance_threshold_data_count
    if imbalance_detected:
        print(f"Data is imbalanced because minority class count {minority_class} < {imbalance_threshold_data_count}  "
              f"is less than {imbalance_threshold * 100}% of the total data {Y_Data.shape[0]}")
    else:
        print(f"Data is balanced because because minority class count {minority_class} > "
              f"{imbalance_threshold_data_count} is greater than {imbalance_threshold * 100}% of the total "
              f"data {Y_Data.shape[0]}")

    # concat two dataframes for better visualization

    for key, item in vector_dict.items():
        print(key, " =========================================================  ")

        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(item[0], item[1], test_size=0.3, random_state=42)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Train-test split: " + str(duration) + " secs")

        start = time.time()
        if imbalance_detected:
            class_weights = get_class_weights(item[1])
            logist_reg = LogisticRegression(class_weight=class_weights)
        else:
            logist_reg = LogisticRegression()
        # create a multi-class classifier one vs rest logistic regression with class weights
        model = multiclass.OneVsRestClassifier(logist_reg)
        scaler = StandardScaler(
            with_mean=False)  # Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
        pipe = make_pipeline(scaler, model)


        if key == 'Scipy sparse matrix':
            y = y_train.toarray().ravel()
            # add classweights
            pipe.fit(X_train, y)
        elif key == 'Sparse pandas dataframe':
            pipe.fit(X_train.values, y_train.values.ravel())
        else:
            pipe.fit(X_train.values, y_train.values.ravel())
        end = time.time()
        duration = round(end - start, 2)
        #        print("Training took : " + str(duration) + " secs")
        #        print("\n")

        # plot confusion matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        start = time.time()
        y_pred = pipe.predict(X_test)
        if key == 'Scipy sparse matrix':
            cm = confusion_matrix(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            cm = confusion_matrix(y_test.values.ravel(), y_pred)
        else:
            cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        end = time.time()
        duration = round(end - start, 2)
        #        print("Confusion matrix took: " + str(duration) + " secs")
        plt.show()
        # plot precision score
        from sklearn.metrics import precision_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            precision_value = precision_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            precision_value = precision_score(y_test.values.ravel(), y_pred)
        else:
            precision_value = precision_score(y_test, y_pred)
        print("Precision score: ", precision_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Precision score took : " + str(duration) + " secs")
        print("""Precision is essentially how much you can trust its ability to label something as positive
The best value is 1 and the worst value is 0.""")
        if precision_value < 0.5:
            print(
                "ERROR: Precision is less than 0.5, which means the model is not good at labeling something as positive.")
        print("\n")

        # plot recall score
        from sklearn.metrics import recall_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            recall_value = recall_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            recall_value = recall_score(y_test.values.ravel(), y_pred)
        else:
            recall_value = recall_score(y_test, y_pred)
        print("Recall score: ", recall_value)
        end = time.time()
        duration = round(end - start, 2)
        print("Recall score: " + str(duration) + " secs")
        print("""Recall is essentially how many of the Actual Positives it found
The best value is 1 and the worst value is 0.""")
        if recall_value < 0.5:
            print("""ERROR: Recall is less than 0.5, which means the model is not good at finding actual positives.
1) Collect more data: More data will help the model learn the patterns better.
2) Change the model: Try a different algorithm or tweak the hyperparameters of the current one.
3) Change the features: Use a different set of features that are more relevant to the problem. 
3.1) Use domain knowledge: Applying domain knowledge to the feature engineering process (i.e., the process of selecting and creating the input features used by the model) can help improve the precision and recall of the model. 
4) Resample the data: If the data is imbalanced, resampling techniques can help balance it out.
5) Change the threshold: The threshold for classification can be changed to favor precision or recall.
6) Implement class weights: If the positive and negative cases in the dataset are imbalanced (e.g., there are significantly more negative cases than positive cases), then the model may be biased towards the more prevalent class. Implementing class weights (i.e., giving more weight to the minority class) can help balance the precision and recall of the model.
7) Use ensembling: Combining multiple models can help improve the overall performance of the model.
8) Use cross-validation: Cross-validation can help evaluate the model's performance more accurately and reduce the risk of overfitting.
9) Use data augmentation: Data augmentation techniques can help increase the amount of training data available to the model, which can improve its performance. This is the process of generating additional training data by applying transformations to the existing data
10) Use data balancing techniques: (e.g., there are significantly more negative cases than positive cases) Data balancing techniques can help address the issue of imbalanced datasets by either oversampling the minority class or undersampling the majority class to create a more balanced dataset. 
""")
        print("\n")

        # plot accuracy score
        from sklearn.metrics import accuracy_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            accuracy_value = accuracy_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            accuracy_value = accuracy_score(y_test.values.ravel(), y_pred)
        else:
            accuracy_value = accuracy_score(y_test, y_pred)
        print("Accuracy score: ", accuracy_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Accuracy score: " + str(duration) + " secs")
        print("""Intuitively, How close it predicts the actual values positive or negative
The best value is 1 and the worst value is 0.""")
        if accuracy_value < 0.5:
            print("ERROR: Accuracy is less than 0.5, which means the model is not good at predicting actual values.")
        print("\n")

        # plot f1 score
        from sklearn.metrics import f1_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            f1_value = f1_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            f1_value = f1_score(y_test.values.ravel(), y_pred)
        else:
            f1_value = f1_score(y_test, y_pred)
        print("F1 score: ", f1_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("F1 score: " + str(duration) + " secs")
        print("""F1 is essentially a weighted average of the true positive rate (recall) and precision
The best value is 1 and the worst value is 0.
        """)
        print("\n")

        # plot balanced accuracy score
        from sklearn.metrics import balanced_accuracy_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            balanced_accuracy_value = balanced_accuracy_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            balanced_accuracy_value = balanced_accuracy_score(y_test.values.ravel(), y_pred)
        else:
            balanced_accuracy_value = balanced_accuracy_score(y_test, y_pred)
        print("Balanced accuracy score: ", balanced_accuracy_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Balanced accuracy score: " + str(duration) + " secs")
        #        print("\n")

        # plot roc auc score
        from sklearn.metrics import roc_auc_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            roc_auc_value = roc_auc_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            roc_auc_value = roc_auc_score(y_test.values.ravel(), y_pred)
        else:
            roc_auc_value = roc_auc_score(y_test, y_pred)
        print("ROC AUC score: ", roc_auc_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("ROC AUC score: " + str(duration) + " secs")
        #        print("\n")

        # plot ROC curve
        from sklearn.metrics import RocCurveDisplay, roc_curve

        start = time.time()
        y_score = pipe.decision_function(X_test)
        if key == 'Scipy sparse matrix':
            fpr, tpr, _ = roc_curve(y_test.toarray().ravel(), y_score)
        elif key == 'Sparse pandas dataframe':
            fpr, tpr, _ = roc_curve(y_test.values.ravel(), y_score)
        else:
            fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        roc_display.plot()
        end = time.time()
        duration = round(end - start, 2)
        #        print("ROC curve: " + str(duration) + " secs")
        plt.show()
        #        print("\n")

        # plot precision recall curve
        from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

        start = time.time()
        if key == 'Scipy sparse matrix':
            precision, recall, _ = precision_recall_curve(y_test.toarray().ravel(), y_score)
        elif key == 'Sparse pandas dataframe':
            precision, recall, _ = precision_recall_curve(y_test.values.ravel(), y_score)
        else:
            precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
        pr_display.plot()
        end = time.time()
        duration = round(end - start, 2)
        #        print("Precision recall curve: " + str(duration) + " secs")
        plt.show()
        #        print("\n")

        # plot classification report
        from sklearn.metrics import classification_report

        start = time.time()
        if key == 'Scipy sparse matrix':
            classification_report_value = classification_report(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            classification_report_value = classification_report(y_test.values.ravel(), y_pred)
        else:
            classification_report_value = classification_report(y_test, y_pred)
        print("classification_report", classification_report_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Classification report: " + str(duration) + " secs")
        #        print("\n")

        # plot log loss
        from sklearn.metrics import log_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            log_loss_value = log_loss(y_test.toarray().ravel(), y_pred)

        elif key == 'Sparse pandas dataframe':
            log_loss_value = log_loss(y_test.values.ravel(), y_pred)
        else:
            log_loss_value = log_loss(y_test, y_pred)
        #        print("Log loss: ", log_loss(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        #        print("Log loss: " + str(duration) + " secs")
        #        print("\n")

        # plot cohen kappa score
        from sklearn.metrics import cohen_kappa_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            cohen_kappa_score_value = cohen_kappa_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            cohen_kappa_score_value = cohen_kappa_score(y_test.values.ravel(), y_pred)
        else:
            cohen_kappa_score_value = cohen_kappa_score(y_test, y_pred)
        print("Cohen kappa score: ", cohen_kappa_score_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Cohen kappa score: " + str(duration) + " secs")
        #        print("\n")

        # plot matthews correlation coefficient
        from sklearn.metrics import matthews_corrcoef

        start = time.time()
        if key == 'Scipy sparse matrix':
            matthews_correlation_coeff_value = matthews_corrcoef(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            matthews_correlation_coeff_value = matthews_corrcoef(y_test.values.ravel(), y_pred)
        else:
            matthews_correlation_coeff_value = matthews_corrcoef(y_test, y_pred)
        print("Matthews correlation coefficient: ", matthews_correlation_coeff_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Matthews correlation coefficient: " + str(duration) + " secs")
        #        print("\n")

        # plot jaccard score
        from sklearn.metrics import jaccard_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            jaccard_score_value = jaccard_score(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            jaccard_score_value = jaccard_score(y_test.values.ravel(), y_pred)
        else:
            jaccard_score_value = jaccard_score(y_test, y_pred)
        print("Jaccard score: ", jaccard_score_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Jaccard score: " + str(duration) + " secs")
        #        print("\n")

        # plot hamming loss
        from sklearn.metrics import hamming_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            hamming_loss_value = hamming_loss(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            hamming_loss_value = hamming_loss(y_test.values.ravel(), y_pred)
        else:
            hamming_loss_value = hamming_loss(y_test, y_pred)
        print("Hamming loss: ", hamming_loss_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Hamming loss: " + str(duration) + " secs")
        #        print("\n")

        # plot zero one loss
        from sklearn.metrics import zero_one_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            zero_one_loss_value = zero_one_loss(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            zero_one_loss_value = zero_one_loss(y_test.values.ravel(), y_pred)
        else:
            zero_one_loss_value = zero_one_loss(y_test, y_pred)
        print("Zero one loss: ", zero_one_loss_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Zero one loss: " + str(duration) + " secs")
        #        print("\n")

        # plot brier score loss
        from sklearn.metrics import brier_score_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            brier_score_loss_value = brier_score_loss(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            brier_score_loss_value = brier_score_loss(y_test.values.ravel(), y_pred)
        else:
            brier_score_loss_value = brier_score_loss(y_test, y_pred)
        print("Brier score loss: ", brier_score_loss_value)
        end = time.time()
        duration = round(end - start, 2)
        #        print("Brier score loss: " + str(duration) + " secs")
        #        print("\n")

        # plot hinge loss
        from sklearn.metrics import hinge_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            hinge_loss_value = hinge_loss(y_test.toarray().ravel(), y_pred)
        elif key == 'Sparse pandas dataframe':
            hinge_loss_value = hinge_loss(y_test.values.ravel(), y_pred)
        else:
            hinge_loss_value = hinge_loss(y_test, y_pred)
        print("Hinge loss: ", hinge_loss_value)
        end = time.time()
        duration = round(end - start, 2)
#        print("Hinge loss: " + str(duration) + " secs")
#        print("\n")
