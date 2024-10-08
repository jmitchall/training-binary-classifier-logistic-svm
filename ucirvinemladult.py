import os
import time

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

    vector_dict = {'Pandas dataframe': [X_Data_one_hot_encoded, Y_Data],
                   'Sparse pandas dataframe': [X_Data_sparse_one_hot_encoded, Y_Data_sparse_one_hot_encoded],
                   'Scipy sparse matrix': [X_Data_csr, Y_Data_csr]
                   }
    #
    for key, item in vector_dict.items():
        print(key)

        start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(item[0], item[1], test_size=0.3, random_state=42)
        end = time.time()
        duration = round(end - start, 2)
        print("Train-test split: " + str(duration) + " secs")

        start = time.time()
        model = multiclass.OneVsRestClassifier(LogisticRegression())
        scaler = StandardScaler(
            with_mean=False)  # Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.
        pipe = make_pipeline(scaler, model)
        if key == 'Scipy sparse matrix':
            y = y_train.toarray().ravel()
            pipe.fit(X_train, y)
        elif key == 'Sparse pandas dataframe':
            pipe.fit(X_train.values, y_train.values.ravel())
        else:
            pipe.fit(X_train.values, y_train.values.ravel())
        end = time.time()
        duration = round(end - start, 2)
        print("Training: " + str(duration) + " secs")
        print("\n")

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
        print("Confusion matrix: " + str(duration) + " secs")
        plt.show()
        print("\n")

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
        print("ROC curve: " + str(duration) + " secs")
        plt.show()
        print("\n")

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
        print("Precision recall curve: " + str(duration) + " secs")
        plt.show()
        print("\n")

        # plot classification report
        from sklearn.metrics import classification_report

        start = time.time()
        if key == 'Scipy sparse matrix':
            print(classification_report(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print(classification_report(y_test.values.ravel(), y_pred))
        else:
            print(classification_report(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Classification report: " + str(duration) + " secs")
        print("\n")

        # plot log loss
        from sklearn.metrics import log_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Log loss: ", log_loss(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Log loss: ", log_loss(y_test.values.ravel(), y_pred))
        else:
            print("Log loss: ", log_loss(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Log loss: " + str(duration) + " secs")
        print("\n")

        # plot accuracy score
        from sklearn.metrics import accuracy_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Accuracy score: ", accuracy_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Accuracy score: ", accuracy_score(y_test.values.ravel(), y_pred))
        else:
            print("Accuracy score: ", accuracy_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Accuracy score: " + str(duration) + " secs")
        print("\n")

        # plot f1 score
        from sklearn.metrics import f1_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("F1 score: ", f1_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("F1 score: ", f1_score(y_test.values.ravel(), y_pred))
        else:
            print("F1 score: ", f1_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("F1 score: " + str(duration) + " secs")
        print("\n")

        # plot balanced accuracy score
        from sklearn.metrics import balanced_accuracy_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Balanced accuracy score: ", balanced_accuracy_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Balanced accuracy score: ", balanced_accuracy_score(y_test.values.ravel(), y_pred))
        else:
            print("Balanced accuracy score: ", balanced_accuracy_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Balanced accuracy score: " + str(duration) + " secs")
        print("\n")

        # plot cohen kappa score
        from sklearn.metrics import cohen_kappa_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Cohen kappa score: ", cohen_kappa_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Cohen kappa score: ", cohen_kappa_score(y_test.values.ravel(), y_pred))
        else:
            print("Cohen kappa score: ", cohen_kappa_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Cohen kappa score: " + str(duration) + " secs")
        print("\n")

        # plot matthews correlation coefficient
        from sklearn.metrics import matthews_corrcoef

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Matthews correlation coefficient: ", matthews_corrcoef(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Matthews correlation coefficient: ", matthews_corrcoef(y_test.values.ravel(), y_pred))
        else:
            print("Matthews correlation coefficient: ", matthews_corrcoef(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Matthews correlation coefficient: " + str(duration) + " secs")
        print("\n")

        # plot roc auc score
        from sklearn.metrics import roc_auc_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("ROC AUC score: ", roc_auc_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("ROC AUC score: ", roc_auc_score(y_test.values.ravel(), y_pred))
        else:
            print("ROC AUC score: ", roc_auc_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("ROC AUC score: " + str(duration) + " secs")
        print("\n")

        # plot precision score
        from sklearn.metrics import precision_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Precision score: ", precision_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Precision score: ", precision_score(y_test.values.ravel(), y_pred))
        else:
            print("Precision score: ", precision_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Precision score: " + str(duration) + " secs")
        print("\n")

        # plot recall score
        from sklearn.metrics import recall_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Recall score: ", recall_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Recall score: ", recall_score(y_test.values.ravel(), y_pred))
        else:
            print("Recall score: ", recall_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Recall score: " + str(duration) + " secs")
        print("\n")

        # plot jaccard score
        from sklearn.metrics import jaccard_score

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Jaccard score: ", jaccard_score(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Jaccard score: ", jaccard_score(y_test.values.ravel(), y_pred))
        else:
            print("Jaccard score: ", jaccard_score(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Jaccard score: " + str(duration) + " secs")
        print("\n")

        # plot hamming loss
        from sklearn.metrics import hamming_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Hamming loss: ", hamming_loss(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Hamming loss: ", hamming_loss(y_test.values.ravel(), y_pred))
        else:
            print("Hamming loss: ", hamming_loss(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Hamming loss: " + str(duration) + " secs")
        print("\n")

        # plot zero one loss
        from sklearn.metrics import zero_one_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Zero one loss: ", zero_one_loss(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Zero one loss: ", zero_one_loss(y_test.values.ravel(), y_pred))
        else:
            print("Zero one loss: ", zero_one_loss(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Zero one loss: " + str(duration) + " secs")
        print("\n")

        # plot brier score loss
        from sklearn.metrics import brier_score_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Brier score loss: ", brier_score_loss(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Brier score loss: ", brier_score_loss(y_test.values.ravel(), y_pred))
        else:
            print("Brier score loss: ", brier_score_loss(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Brier score loss: " + str(duration) + " secs")
        print("\n")

        # plot hinge loss
        from sklearn.metrics import hinge_loss

        start = time.time()
        if key == 'Scipy sparse matrix':
            print("Hinge loss: ", hinge_loss(y_test.toarray().ravel(), y_pred))
        elif key == 'Sparse pandas dataframe':
            print("Hinge loss: ", hinge_loss(y_test.values.ravel(), y_pred))
        else:
            print("Hinge loss: ", hinge_loss(y_test, y_pred))
        end = time.time()
        duration = round(end - start, 2)
        print("Hinge loss: " + str(duration) + " secs")
        print("\n")
