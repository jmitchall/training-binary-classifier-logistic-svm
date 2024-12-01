import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import multiclass
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
    :param top_k:  number of best features to select
    :return:
    """
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    x_df = df_full_data.iloc[:, 0:df_full_data.shape[1] - 1]
    y_df = df_full_data.iloc[:, df_full_data.shape[1] - 1]
    best_features = SelectKBest(score_func=chi2, k=top_k)
    fit = best_features.fit(x_df, y_df)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(x_df.columns)
    # concat two dataframes for better visualization
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']
    return feature_scores.nlargest(top_k, 'Score')


def select_feature_importance(df_full_data, top_k=10):
    """
    Selects the K best features of the data frame using chi2
    2. Feature Importance
       You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
    :param df_full_data: must be a pandas data frame that has the target column as the last column
    :param top_k:  number of best features to select
    :return:
    """
    from sklearn.ensemble import ExtraTreesClassifier
    df_x = df_full_data.iloc[:, 0:df_full_data.shape[1] - 1]
    df_y = df_full_data.iloc[:, df_full_data.shape[1] - 1]
    tree_model = ExtraTreesClassifier()
    tree_model.fit(df_x, df_y)
    feat_importances = pd.Series(tree_model.feature_importances_, index=df_x.columns)
    feat_importances.nlargest(top_k).plot(kind='barh')

    # return top_k features
    return feat_importances.nlargest(top_k)


def select_correlation_features(df_full_data):
    """
    Selects the K best features of the data frame using chi2
    3. Correlation Matrix with Heatmap
       Correlation states how the features are related to each other or the target variable.
    :param df_full_data: must be a pandas data frame that has the target column as the last column
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
    label_classes = df_classes.iloc[:, df_classes.shape[1] - 1]
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(label_classes), y=label_classes)
    # convert Class weights to dictionary
    class_weights = dict(enumerate(class_weights))
    return class_weights


ONE_HOT_MISSING_VALUE = 0.0


def get_svc_model(imbalanced, class_data, inverse_of_regularization_strength=1.0,kernel_type='rbf', gamma='scale'):
    """
    The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of
    thousands of samples. For large datasets consider using LinearSVC or SGDClassifier instead,
    possibly after a Nystroem transformer or other Kernel Approximation.

    The multiclass support is handled according to a one-vs-one scheme.

    For details on the precise mathematical formulation of the provided kernel functions and how gamma,
    coef0 and degree affect each other, see the corresponding section in the narrative documentation:
    https://scikit-learn.org/dev/modules/svm.html#svm-kernels
    https://scikit-learn.org/dev/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py
    :param imbalanced:
    :param class_data:
    :param inverse_of_regularization_strength:
    :param kernel_type:
    :param gamma:
    :return:
    """
    if imbalanced:
        class_weights = get_class_weights(class_data)
        return_value = SVC(C=inverse_of_regularization_strength, class_weight=class_weights, kernel=kernel_type, gamma=gamma)
    else:
        return_value = SVC(C=inverse_of_regularization_strength , kernel=kernel_type, gamma=gamma)
    return multiclass.OneVsRestClassifier(return_value)


def is_sparse_dataframe(data_df):
    """
    Checks if the data frame is a sparse data frame
    :param data_df: pandas data frame
    :return: boolean
    """
    # if data_df is a csr_matrix return False
    if hasattr(data_df, 'data'):
        return False

    # return true if any data_df[col] is a pandas.arrays.SparseArray
    return any(isinstance(data_df[col].dtype, pd.SparseDtype) for col in data_df.columns)


def return_flattened_data(data_df):
    """
    Converts the data frame to a flattened data frame
    :param data_df: pandas data frame
    :return: pandas data frame
    """
    # if the data_df is a Scipy sparse matrix convert it to a Dense data frame using .toarray() and
    # then return .ravel()
    # if the data_df is a sparse data frame return the data_df as a Dense data frame using .values.ravel()
    # else return the data_df
    if isinstance(data_df, sparse.csr_matrix):
        return data_df.toarray().ravel()
    elif is_sparse_dataframe(data_df):
        return data_df.values.ravel()
    else:
        return data_df


def imbalanced_data(y_test_imbalanced, y_pred_imbalanced, y_score_imbalanced):
    # plot accuracy score
    from sklearn.metrics import accuracy_score
    accuracy_value = accuracy_score(y_test_imbalanced, y_pred_imbalanced)
    print("Accuracy score: ", accuracy_value)
    print("Intuitively, How close it predicts the actual values positive or negative "
          "Accuracy score is a good measure when the dataset is balanced, "
          "meaning there are similar numbers of examples in each class. "
          "The best value is 1 and the worst value is 0.")
    if accuracy_value < 0.5:
        print("ERROR: Accuracy is less than 0.5, which means the model "
              "is not good at predicting actual values.")
    print("\n")
    # plot roc auc score
    from sklearn.metrics import roc_auc_score
    roc_auc_value = roc_auc_score(y_test_imbalanced, y_pred_imbalanced)
    print("ROC AUC score: ", roc_auc_value)
    print("ROC AUC a metric that summarizes how well a classifier can distinguish\n"
          "between positive and negative classes. It is calculated by measuring the area under the ROC curve,"
          "\n"
          "which plots the True Positive Rate (TPR) on the y-axis and the False Positive Rate (FPR) on the"
          " x-axis.\n\n"
          "ROC Curves summarize the trade-off between the true positive rate and false positive rate for a\n"
          "predictive model using different probability thresholds.\n"
          "\n"
          "ROC curves are appropriate when the observations are balanced between each class\n"
          "\n"
          "A higher ROC AUC score indicates better performance. A score of 1.0 indicates a perfect model,\n"
          "while a score of 0.5 indicates a random model.\n"
          "\n"
          "The ROC curve shows how well a model works for every possible threshold. The ROC curve is based\n"
          "on the TPR and FPR, which are derived from a confusion matrix. The confusion matrix compares \n"
          "predicted values against actual values.\n"
          ""
          "When Comparing two models, the model with the higher ROC AUC score is better. "
          )
    # plot ROC curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, roc_curve
    fpr, tpr, _ = roc_curve(y_test_imbalanced, y_score_imbalanced.ravel())
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()
    plt.legend(["roc curve"], fontsize="x-large")
    plt.show()
    # plot cohen kappa score
    from sklearn.metrics import cohen_kappa_score
    cohen_kappa_score_value = cohen_kappa_score(y_test_imbalanced, y_pred_imbalanced)
    print("Cohen kappa score: ", cohen_kappa_score_value)
    print("Cohen’s kappa is a metric that measures inter-annotator agreement. "
          "to quantify the agreement between raters, judges, or observers. used to compare how different raters "
          "classify items into categories. It's particularly useful when the categories are nominal, "
          "meaning they don't have a natural order"
          "It considers how much better the agreement is over and above chance. "
          "1: Perfect agreement between raters "
          "0: Observed agreement is the same as chance agreement "
          "Negative: There is less agreement than random chance "
          "Negative scores from Cohen's kappa can indicate raters with different viewpoints. Disagreements"
          "Cohen's kappa is a statistical measure that indicates how well two"
          "raters agree when classifying items into categories."
          "Cohen's kappa can measure how well a model agrees with human ratings. "
          "It's especially helpful when dealing with imbalanced data, "
          "where overall accuracy can be misleading. Measure agreement about datasets that are imbalanced or "
          "where random guessing could lead to a high accuracy rate."
          "It is important to note that the Cohen's Kappa coefficient can only tell you how reliably both raters "
          "are measuring the same thing. It does not tell you whether what the two raters are measuring "
          "is the right thing!"
          "https://datatab.net/tutorial/cohens-kappa "
          )
    # plot hamming loss
    from sklearn.metrics import hamming_loss
    hamming_loss_value = hamming_loss(y_test_imbalanced, y_pred_imbalanced)
    print("Hamming loss: ", hamming_loss_value)
    print("""Hamming loss is used in multilabel classification.\n
            accuracy measures the overall proportion of correct predictions, \n
            while Hamming loss specifically focuses on the fraction of incorrectly\n
            predicted labels in multi-label classification tasks\n
            HL=1-Accuracy
            Hamming loss is the fraction of labels that are incorrectly predicted. 
            The best value is 0 and the worst value is 1.
            It is a loss function used for multilabel classification, 
            where the model predicts multiple labels for each instance. 
            Used to measure how often a learning algorithm incorrectly predicts the 
            relevance of an example to a class label\n
            measures the average number of times a model incorrectly predicts the relevance of \n
            an example to a class label. It takes into account both \n
            prediction errors (incorrect labels) and \n
            missing errors (relevant labels not predicted)\n
            It penalizes individual labels, which is more forgiving than the subset zero-one loss. 
            A smaller Hamming loss value indicates better performance by the learning algorithm\n
            Hamming loss is calculated by performing an exclusive or (XOR) between the predicted 
            and actual labels, and then averaging across the dataset.
            Hamming loss and Hamming distance are the same thing, but they are used in different contexts.
            hamming loss is calculated as the hamming distance between y_true and y_pred 
            Hamming Distance refers to the number of positions at which two strings of the same length differ. 
            It is a metric used in computer science to measure dissimilarity between strings.""")
    # plot log loss for Statistical models
    from sklearn.metrics import log_loss
    log_loss_value = log_loss(y_test_imbalanced, y_pred_imbalanced)
    print("Log loss: ", log_loss(y_test_imbalanced, y_pred_imbalanced), "log loss value: ", log_loss_value)
    print("Log loss is a measure of how well a model predicts the probabilities of the positive class.\n"
          "og-loss is indicative of how close the prediction probability is to the corresponding\n"
          "actual/true value (0 or 1 in case of binary classification). The more the predicted probability\n"
          "diverges from the actual value, the higher is the log-loss value."
          "\n"
          " 0 is the best value, and values closer to 0 are better. The worst value is infinity."
          "\n"
          "Like ROC-AUC score Log-loss is one of the major metrics to assess the performance of a\n"
          "classification problem. it is this prediction probability of a data record that the log-loss\n"
          "value is dependent on.\n"
          "\n"
          "The Lowest log-loss score fore the data set is regarded as the baseline log-loss score\n"
          "and the model is expected to perform better than this score.\n"
          "\n"
          "Baseline log-loss score for a dataset is determined from the naïve classification model,\n"
          "which simply pegs all the observations with a constant probability equal to % of data \n"
          "with class 1 observations.\n\n"
          "Higher the imbalance in a dataset, lower the baseline log-loss score of the "
          "dataset, due to lower proportion of observations in the minority class. "
          "\n"
          "log-loss values should always be interpreted in context of the baseline score as "
          "provided by the naïve model.\n"
          "\n"
          "When we build a statistical model on a given dataset, the model must beat the baseline log-loss score\n"
          "thereby proving itself to be more skillful than the naïve model. If that does not turn out to be the\n"
          " case, it implies that the trained statistical model is not helpful at all, and it would be better \n"
          "to go with the naïve model instead"
          "https://towardsdatascience.com/intuition-behind-log-loss-score-4e0c9979680a"
          "Alternative to Zero-one. For binary classification, this provides a smooth, differentiable "
          "approximation to the zero-one loss, allowing for efficient gradient-based optimization"
          "log loss is more sensitive to differences in predicted probabilities and can be used as an "
          "objective function for training machine learning models")


def balanced_data(y_test_balanced, y_pred_balanced, y_score_balanced):
    # plot balanced accuracy score
    import matplotlib.pyplot as plt
    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy_value = balanced_accuracy_score(y_test_balanced, y_pred_balanced)
    print("Balanced accuracy score: ", balanced_accuracy_value)
    print("Balanced accuracy is the arithmetic mean of sensitivity and specificity. "
          "A metric that measures the average accuracy of a model across "
          "both the minority and majority classes. The best value is 1 and the worst value is 0.")
    # plot precision recall curve
    from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
    precision, recall, _ = precision_recall_curve(y_test_balanced, y_score_balanced)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_display.plot()
    plt.legend(["precision-recall curve "], fontsize="x-large")
    plt.show()
    print("The precision-recall curve is used for evaluating the performance of binary "
          "classification algorithms. It is often used in situations where classes "
          "are heavily imbalanced. ")
    # plot average precision score
    from sklearn.metrics import average_precision_score
    average_precision_value = average_precision_score(y_test_balanced, y_score_balanced)
    print("Average precision score: ", average_precision_value)
    print("The average precision score summarizes the precision-recall curve as the weighted mean of "
          "precisions\nachieved at each threshold, with the increase in recall from the previous \n"
          "threshold used as the weight. The best value is 1 and the worst value is 0."
          "https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248")
    # plot zero one loss
    from sklearn.metrics import zero_one_loss
    zero_one_loss_value = zero_one_loss(y_test_balanced, y_pred_balanced)
    print("Zero one loss: ", zero_one_loss_value)
    print("Zero-one loss, also known as 0/1 loss, is a loss function used to evaluate classifiers "
          "in multi-class/binary classification settings. "
          "It is defined as the fraction of incorrect predictions made by the model. "
          "Zero-one loss counts the number of mistakes a classifier makes on a training set. "
          "It assigns a loss of 1 for each incorrectly predicted example and 0 for each correct prediction"
          "The best value is 0 and the worst value is 1."
          "not often used for optimization \n"
          "Zero-one loss is rarely used to guide optimization procedures because "
          "it's non-continuous meaning is a type of optimization problem where the graph of the function"
          " contains one or more breaks. it is also non-differentiable. which means that it is not "
          "possible to calculate the gradient of the loss function with respect to the model parameters.\n"
          "or a variety of reasons functions are non-convex. The functions in this class of optimization are "
          "generally non-smooth. These functions often contain sharp points or corners that do not allow for "
          "the solution of a tangent and are thus non-differentiable."
          "Zero-one loss is robust to outliers(  is less affected by extreme values, or outliers, in a data set)"
          "because it's not affected by how far a misclassified point is from the margin."
          "works well with datasets where the primary concern is simply \n"
          "identifying the correct class for each data point, meaning it is most suitable for \n"
          "binary classification problems with well-defined, \n"
          "distinct classes where the cost of miss classification is considered equal for "
          "both positive and negative predictions;  "
          "To identify areas where a model is struggling with specific classes or data points, "
          "When you want a straightforward measure of classification accuracy to compare different "
          "models on the same dataset. ")
    # plot f1 score
    from sklearn.metrics import f1_score
    f1_value = f1_score(y_test_balanced, y_pred_balanced)
    print("F1 score: ", f1_value)
    print("""F1 is essentially a weighted average of the true positive rate (recall) and precision
            The best value is 1 and the worst value is 0.
            """)
    print("\n")
    # plot classification report
    from sklearn.metrics import classification_report
    classification_report_value = classification_report(y_test_balanced, y_pred_balanced)
    print("classification_report\n", classification_report_value)
    # plot matthews correlation coefficient
    from sklearn.metrics import matthews_corrcoef
    matthews_correlation_coeff_value = matthews_corrcoef(y_test_balanced, y_pred_balanced)
    print("Matthews correlation coefficient: ", matthews_correlation_coeff_value)
    print("The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary\n"
          "and multiclass classifications. It takes into account true and false positives and negatives and is\n"
          "generally regarded as a balanced measure which can be used even if the classes are of very different\n"
          "sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of\n"
          "+1 represents a perfect prediction,\n"
          " 0 an average random prediction and\n"
          "-1 an inverse prediction. \n"
          "The statistic is also known as the phi coefficient. The MCC is a useful measure even if the classes are of\n"
          "very different sizes. It is considered to be a balanced measure, as opposed to the F1 score, which is\n"
          "not. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1\n"
          "represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.\n\n"
          ""
          "MCC is more reliable than other metrics, like accuracy and F1 score, \n"
          "which can be misleading on imbalanced datasets\n"
          "MCC produces a high score only when all four basic rates \n"
          "(sensitivity, specificity, precision, and negative predictive value) are high.\n"
          "MCC considers all possible outcomes, including correct and incorrect predictions.\n"
          "MCC provides a balanced approach by considering \n"
          "true positives, true negatives, false positives, and false negatives.\n"
          "https://en.wikipedia.org/wiki/Matthews_correlation_coefficient\n"
          " F-score, the Jaccard similarity coefficient or Matthews' correlation coefficient (MCC), "
          "are not robust to class imbalance in the sense that if the proportion of the minority class tends "
          "to 0, the true positive rate (TPR) of the Bayes classifier under these metrics tends to 0 as well. "
          "Thus, in imbalanced classification problems, these metrics favour classifiers which ignore the "
          "minority class. "
          "https://arxiv.org/abs/2404.07661#:~:text=We%20show%20that%20established%20performance,rate%20(TPR)%20of%20the%20Bayes")
    # plot brier score loss
    from sklearn.metrics import brier_score_loss
    brier_score_loss_value = brier_score_loss(y_test_balanced, y_pred_balanced)
    print("Brier score loss: ", brier_score_loss_value)
    print("Brier score loss is a loss function used in binary and multiclass classification tasks."
          "It evaluates the accuracy of probabilistic predictions. "
          "It is a measure of the mean squared difference between the predicted probabilities and the actual "
          "outcomes. It is applicable to tasks where predictions must assign probabilities to "
          "a set of mutually exclusive discrete outcomes or classes."
          "The Brier score is a proper score function that measures the accuracy of probabilistic predictions. "
          "It is the mean squared difference between the predicted probabilities and the actual outcomes."
          "The Brier score ranges from 0 to 1, with \n"
          "0 indicating perfect predictions and 1 indicating perfectly wrong predictions.\n "
          "The best value is 0 and the worst value is 1."
          "Well-suited for imbalanced datasets The Brier score focuses on the "
          "probabilities of the positive (usually minority) class. "
          "The Brier score is a measure of the accuracy of probabilistic predictions. "
          "It is calculated as the mean squared difference between the predicted probabilities and the actual "
          "outcomes. The Brier score should not be used alone for comparing model performance, as it does not  "
          "take into account the relative costs of false positives and false negatives. "
          "The Brier score is a proper scoring rule, meaning that it is optimized when the predicted probabilities "
          "are calibrated. A calibrated model is one where the predicted probabilities accurately reflect the "
          "true probabilities of the outcomes and yields consistent probabilistic predictions."
          "The Brier score is a useful metric for evaluating the performance of probabilistic classifiers. "
          "It can be used to compare the accuracy of different models and to assess the calibration of predicted "
          "probabilities. "
          "https://en.wikipedia.org/wiki/Brier_score")
    # plot jaccard score
    from sklearn.metrics import jaccard_score
    jaccard_score_value = jaccard_score(y_test_balanced, y_pred_balanced)
    print("Jaccard score: ", jaccard_score_value)
    print("Jaccard score is a measure of similarity between two sets."
          "The Jaccard index, also known as the Jaccard similarity coefficient, "
          "Jaccard Similarity = (number of observations in both sets) / (number in either set)"
          "Jaccard Distance = 1 — Jaccard Similarity = measures the dissimilarity between two sets"
          "is defined as the size of the intersection divided by the size of the union of two sets. "
          "The best value is 1 and the worst value is 0."
          "Jaccard similarity is unaffected by the size of the sets being compared. "
          "It is a useful metric for comparing the similarity of two sets,"
          "Jaccard similarity can be used to compare many types of data, \n"
          "including text, images, photos, and time series data. "
          "Jaccard similarity is effective for binary attributes, such as presence or absenc "
          "Jaccard similarity considers a unique set of words for each sentence, so repeating words "
          "in a sentence doesn't change the similarity score. It ignores term frequency."
          "It may not be the best solution for benchmarking."
          "It may be less effective for high-dimensional data. "
          "it ranges from 0 to 1, with \n"
          "1 stating the two groups are identical, and \n"
          "0 indicating there are no shared members "
          "https://medium.com/@mayurdhvajsinhjadeja/jaccard-similarity-34e2c15fb524"
          " F-score, the Jaccard similarity coefficient or Matthews' correlation coefficient (MCC), "
          "are not robust to class imbalance in the sense that if the proportion of the minority class tends "
          "to 0, the true positive rate (TPR) of the Bayes classifier under these metrics tends to 0 as well. "
          "Thus, in imbalanced classification problems, these metrics favour classifiers which ignore the "
          "minority class. "
          "https://arxiv.org/abs/2404.07661#:~:text=We%20show%20that%20established%20performance,rate%20(TPR)%20of%20the%20Bayes")


def get_prediction_score(x_test_data, y_test_data, pipe_model):
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    y_prediction = pipe_model.predict(x_test_data.values)
    cm = confusion_matrix(y_test_data, y_prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.legend(["confusion_matrix"], fontsize="x-large")
    plt.show()
    # plot precision score
    from sklearn.metrics import precision_score
    precision_value = precision_score(y_test_data, y_prediction)
    print("Precision score: ", precision_value)
    print("Precision is essentially how much you can trust its ability to label something as positive\n"
          "the proportion of correct positive predictions made by the model\n"
          "The best value is 1 and the worst value is 0.")
    if precision_value < 0.5:
        print(
            "ERROR: Precision is less than 0.5, which means the model is not good at labeling "
            "something as positive. To increase this try;"
            """
                Precision and recall are a tradeoff. Typically to increase precision for a 
                given model implies lowering recall, though this depends on the precision-recall 
                curve of your model, so you may get lucky. 

                If you want higher precision you need to restrict the positive predictions 
                to those with highest certainty in your model, which means predicting fewer positives overall 
                (which, in turn, usually results in lower recall).
                https://stats.stackexchange.com/questions/186182/a-way-to-maintain-classifiers-recall-while-improving-precision

                If you want to maintain the same level of recall while improving precision, you will need a better classifier.
             """)
    print("\n")
    # plot recall score
    from sklearn.metrics import recall_score
    recall_value = recall_score(y_test_data, y_prediction)
    print("Recall score: ", recall_value)
    print("Recall (aka Sensitivity) is essentially how many of the Actual Positives it found\n"
          "The best value is 1 and the worst value is 0.")
    if recall_value < 0.5:
        print("""ERROR: Recall is less than 0.5, which means the model is not good at finding actual positives.
                    Precision and recall are a tradeoff. Typically to increase precision for a given model implies lowering recall, though this depends on the precision-recall curve of your model, so you may get lucky.
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
    return y_prediction, pipe_model.decision_function(x_test_data.values)


def detect_imbalanced_labels(y_data, imbalance_threshold=0.15):
    # if minority class is less than 15% of the total data, then the data is imbalanced
    minority_class = y_data.value_counts().min()
    imbalance_threshold_data_count = imbalance_threshold * y_data.shape[0]
    imbalance = minority_class < imbalance_threshold_data_count
    if imbalance:
        print(f"Data is imbalanced because minority class count {minority_class} < {imbalance_threshold_data_count}  "
              f"is less than {imbalance_threshold * 100}% of the total data {y_data.shape[0]}"
              "the statistical / probabilistic arithmetic gets quite ugly, quite quickly, with unbalanced data."
              "Solving unbalanced data is basically intentionally biasing your data to get interesting results "
              "instead of accurate results. All methods are vulnerable although SVM and logistic regressions "
              "tend to be a little less vulnerable while decision trees are very vulnerable.\n"
              "I DON'T CARE) You are purely interested in accurate prediction and you think your data is "
              "representative.\nIn this case you do not have to correct at all\n "
              "I DO CARE) Interested in Prediction, You know your source is balanced but your current data is not.\n"
              "Correction needed.\n"
              "I care about rare cases and I want to make sure rare cases are predicted accurately.\n"
              "data imbalance is a problem if \n"
              "a) your model is misspecified, and \n"
              "b) you're either\n "
              "interested in good performance on a minority class or "
              "interested in the model itself. Boosting algorithms ( e.g AdaBoost, XGBoost,…), "
              "because higher weight is given to the minority class at each successive iteration. "
              "during each interation in training the weights of misclassified classes are adjusted."
              " other effective methods are: \n"
              "1) Resampling techniques: Oversampling the minority class or undersampling the majority class. "
              "2) Synthetic data generation: SMOTE (Synthetic Minority Over-sampling Technique) "
              "3) Cost-sensitive learning: Assigning higher costs to misclassifications of the minority class. "
              "4) Anomaly detection: Identifying outliers in the minority class. "
              "5) Ensemble methods: Combining multiple models to improve performance. "
              "6) Transfer learning: Using knowledge from a related task to improve performance. "
              "7) Active learning: Selecting the most informative samples for labeling. "
              "8) Semi-supervised learning: Using a combination of labeled and unlabeled data. "
              "9) Clustering: Grouping similar instances together. "
              "10) Feature selection: Identifying the most relevant features. "
              "11) Data augmentation: Increasing the size of the training set. "
              "12) Model evaluation: Using appropriate metrics to evaluate performance. "
              "13) Model interpretation: Understanding how the model makes predictions. "
              )
    else:
        print(f"Data is balanced because because minority class count {minority_class} > "
              f"{imbalance_threshold_data_count} is greater than {imbalance_threshold * 100}% of the total "
              f"data {y_data.shape[0]}"
              f"IT IS BEST FOR PREDICTION: If you are purely interested in accurate prediction and you "
              f"think your data is representative, then you do not have to correct at all. "
              f"Many classical models simplify neatly under the assumption of balanced data, especially for "
              f"methods like ANOVA that are closely related to experimental design—a traditional / original "
              f"motivation for developing statistical methods"
              f"https://stats.stackexchange.com/questions/283170/when-is-unbalanced-data-really-a-problem-in-machine-learning")
    return imbalance


def get_numeric_data_with_labels(x_data_1_hot_encoded, y_data_labels):
    # select only the numeric columns for the one hot encoded data and then concatenate with the Y_Data
    # to get the top 56 features
    numeric_columns = x_data_1_hot_encoded.select_dtypes(
        include=['int64', 'float64', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16']).columns
    numeric_data = pd.concat([x_data_1_hot_encoded[numeric_columns], y_data_labels], axis=1)
    return numeric_data


def get_standard_scaler(x_data):
    # Standardize features by removing the mean and scaling to unit variance
    # Cannot center sparse matrices: pass `with_mean=False` instead.
    if is_sparse_dataframe(x_data):
        return StandardScaler(with_mean=False)
    else:
        return StandardScaler()


def assign_x_y_pipe_to_dict(model_map, model_key, x_data, y_data, model_predictor,
                            with_pca_components=None, scaler=None):
    simple_imputer = None
    # Set a Flag if the data has null values
    if x_data.isnull().values.any():
        missing_value = ONE_HOT_MISSING_VALUE
        simple_imputer = SimpleImputer(strategy='constant', fill_value=missing_value)
        print("Data has null values assuming one hot encoding, impute with missing value: ", missing_value)
    else:
        print("Data has no null values")

    steps = []
    if not scaler:
        scaler = get_standard_scaler(x_data)
    steps.append(scaler)

    if with_pca_components:
        steps.append(with_pca_components)

    if simple_imputer:
        steps.insert(0, simple_imputer)
    steps.append(model_predictor)
    pipe = make_pipeline(*steps)
    model_map[model_key] = [x_data, y_data, pipe]


def get_range_svc_C(ten_to_minimum_exponent=-4, ten_to_max_exponent=4, num_values=5):
    """
    This function generates a range of values for the regularization strength parameter 'C' in Logistic Regression.
    A high value of C (e.g. C=1.0) "Trust this training data a lot" tells the model to give high weight to the
    training data, and
    a low value (e.g. C=0.01) tells the model to give more weight to this complexity penalty preventing
    overfitting to the training data.
    "This data may not be fully representative of the real world data, so if it's telling you to
    make a parameter really large, don't listen to it".
    C: float, default=1.0 Inverse of regularization strength; must be a positive float.
    Like in support vector machines, smaller values specify stronger regularization.
    Regularization generally refers the concept that there should be a complexity penalty for more extreme parameters.

    :param ten_to_minimum_exponent:
    :param ten_to_max_exponent:
    :param num_values:
    :return: a list of floats containing the regularization strength values
    """
    # The np.logspace function creates a sequence of numbers that are evenly spaced on a logarithmic scale.
    # np.logspace generates numbers spaced evenly on a log scale.
    # In this case, it generates 50 values ranging from ( 10^{ten_to_minimum_exponent} ) to ( 10^{ten_to_max_exponent} ).
    # This array is used to explore different regularization strengths during model training.
    return np.logspace(ten_to_minimum_exponent, ten_to_max_exponent, num_values)


def get_range_svc_kernel():
    """
    This function generates a range of values for the kernel parameter in Support Vector Classifier.
    The kernel parameter specifies the kernel type to be used in the algorithm.
    The kernel function transforms the input data into the required form.
    :return: a list of strings containing the kernel values
    """
    return ['linear', 'poly', 'rbf', 'sigmoid']


def get_range_PCA_components(training_data):
    """
    Principal Component Analysis requires a parameter 'n_components' to be optimised.
    'pca_n_components' signifies the number of components to keep after reducing the dimension.
    The number of components to keep is a hyperparameter that can be tuned to improve the model's performance.

     :param training_data: The training data to be used for PCA
     :return: a list of integers starting from 1 up to the number of columns in the training_data DataFrame
    """
    return list(range(1, training_data.shape[1] + 1, 1))


def find_pipe_with_best_hyper_parameters_grid_search_cross_validation(x_numeric_df, y_data_labels,
                                                                      cross_validation_folds=5, cpu_cores=-1):
    """
    This function finds the best hyperparameters for the model using GridSearchCV.
    GridSearchCV is a meta-estimator that performs cross-validated grid-search over a parameter grid.
    GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”,
    “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the
    estimator used. The parameters of the estimator used to apply these methods are optimized by cross-validated
    grid-search over a parameter grid.
    :param x_numeric_df: The training data to be used for PCA
    :param y_data_labels: The labels for the training data
    :param cross_validation_folds: The number of folds to use for cross-validation
    :param cpu_cores: The number of CPU cores to use for parallel processing
    :return: a tuple containing the GridSearchCV object, a boolean indicating whether the data is imbalanced,
    and a list containing the training data, labels, and the best estimator from the GridSearchCV object
    """
    vector_dict = dict()
    # The parameters are numbers that tells the model what to do with the features, while
    # hyperparameters tell the model how to choose parameters.

    pca_n_components = get_range_PCA_components(x_numeric_df)
    svc_Cs = get_range_svc_C()
    svc_kernels = get_range_svc_kernel()
    # create a range of 1 to 200  in steps of 10
    gammas = [ 'scale', 'auto']
    gammas.extend([1, 0.1, 0.01, 0.001, 0.0001])
    # check for imbalance in the data
    is_imbalance_detected = detect_imbalanced_labels(y_data_labels)
    assign_x_y_pipe_to_dict(vector_dict, f'logit_model model using {x_numeric_df.columns} including PCA',
                            x_numeric_df, y_data_labels,
                            get_svc_model(is_imbalance_detected, class_data=y_data_labels),
                            with_pca_components=PCA())
    hyper_parameter_test_pipe = vector_dict[f'logit_model model using {x_numeric_df.columns} including PCA'][2]
    hyper_parameter_ranges = dict()
    for key, val in hyper_parameter_test_pipe.steps:
        # standardscaler StandardScaler()
        # pca PCA()
        if key == 'pca':
            hyper_parameter_ranges[f"{key}__n_components"] = pca_n_components
        # onevsrestclassifier OneVsRestClassifier(estimator=SVC())
        if key == 'onevsrestclassifier':
            hyper_parameter_ranges[f"{key}__estimator__C"] = svc_Cs
            hyper_parameter_ranges[f"{key}__estimator__kernel"] = svc_kernels
            hyper_parameter_ranges[f"{key}__estimator__gamma"] = gammas

    print("hyper_parameter_ranges: ", hyper_parameter_ranges)
    # GridSearchCV is a meta-estimator that performs cross-validated grid-search over a parameter grid.
    # GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”,
    # “predict_proba”, “decision_function”, “transform” and “inverse_transform”
    # if they are implemented in the estimator used.
    # The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search
    # over a parameter grid.
    gs_cross_validation = GridSearchCV(hyper_parameter_test_pipe, hyper_parameter_ranges, cv=cross_validation_folds,
                                       n_jobs=cpu_cores)
    gs_cross_validation.fit(x_numeric_df, y_data_labels)
    return gs_cross_validation, is_imbalance_detected, [x_numeric_df, y_data_labels,
                                                        gs_cross_validation.best_estimator_]


# python main entry
if __name__ == '__main__':
    adult_data = pd.read_csv('adult.csv', sep=",")
    all_Data_df = pd.read_csv('adult_all_data_processed.csv', sep=",")
    all_Data_mean_encoded_df = pd.read_csv('adult_all_data_target_encoded.csv', sep=",")
    X_Data_label_encoded = pd.read_csv('adult_X_columns_label_encoded.csv', sep=",")
    X_Data_one_hot_encoded = pd.read_csv('adult_X_data_one_hot_encoded.csv', sep=",")
    X_Data_sparse_one_hot_encoded = convert_to_sparse_pandas(
        pd.read_csv('adult_X_sparse_one_hot_encoded.csv', sep=","), [])
    X_Data_csr = sparse.load_npz('adult_X_sparse_one_hot_encoded.npz')
    print('X Data one hot encoded takes up', get_memory_usage_of_data_frame(X_Data_one_hot_encoded))
    print('X sparse Data takes up', get_memory_usage_of_data_frame(X_Data_sparse_one_hot_encoded))
    print('X csr Data takes up', get_csr_memory_usage(X_Data_csr))

    Y_Data_binary = pd.read_csv('adult_Y_data_encode_binary.csv', sep=",")
    Y_Data = pd.read_csv('adult_Y_sparse_one_hot_encoded.csv', sep=",")
    Y_Data_sparse_one_hot_encoded = convert_to_sparse_pandas(Y_Data, [])
    Y_Data_csr = sparse.load_npz('adult_Y_sparse_one_hot_encoded.npz')
    print('Y Data takes up', get_memory_usage_of_data_frame(Y_Data))
    print('Y sparse. Data takes up', get_memory_usage_of_data_frame(Y_Data_sparse_one_hot_encoded))
    print('Y csr Data takes up', get_csr_memory_usage(Y_Data_csr))

    # check for Y_Data imbalance
    print("Y_Data value counts:\n", Y_Data.value_counts())
    print("Y_Data sparse one hot encoded value counts:\n", Y_Data_sparse_one_hot_encoded.value_counts())
    print("Y_Data csr value counts:\n", pd.Series(Y_Data_csr.toarray().ravel()).value_counts())

    numeric_df = get_numeric_data_with_labels(X_Data_one_hot_encoded, Y_Data)

    vector_dict = dict()

    gs_cross_validation_inst, imbalance_detected, best_x_y_pipe = (
        find_pipe_with_best_hyper_parameters_grid_search_cross_validation(
        numeric_df, Y_Data, cross_validation_folds=5, cpu_cores=6))

    print("Best parameters: ", gs_cross_validation_inst.best_params_)
    print("Best score: ", gs_cross_validation_inst.best_score_)
    print("Best estimator: ", gs_cross_validation_inst.best_estimator_)
    print("Best index: ", gs_cross_validation_inst.best_index_)
    vector_dict[f'Best estimator {gs_cross_validation_inst.best_estimator_}'] = best_x_y_pipe

    # Get Logistic Regression model with more iterations as opposed to default 100
    logit_model = get_svc_model(imbalance_detected, class_data=Y_Data,
                                max_iterations=gs_cross_validation_inst.best_params_[
                                                    'onevsrestclassifier__estimator__max_iter'],
                                inverse_of_regularization_strength=
                                                gs_cross_validation_inst.best_params_[
                                                    'onevsrestclassifier__estimator__C'],
                                penalty=gs_cross_validation_inst.best_params_[
                                                    'onevsrestclassifier__estimator__penalty'])

    top_range = 56
    spec_column, score_column = select_k_best_features(numeric_df, top_range).values.T  # transpose the values

    assign_x_y_pipe_to_dict(vector_dict, f'Pandas onehot top {top_range} {spec_column}',
                            X_Data_one_hot_encoded[spec_column], Y_Data, logit_model)

    # Select the important features and check if the k best features detected are all in the important features
    important_features = select_feature_importance(numeric_df, top_range)
    # does the important features contain all the spec_column values
    if not all(elem in important_features.index.values for elem in spec_column):
        # display the difference
        print("Excluded Features : ", set(spec_column) - set(important_features.index.values))
        important_columns = list(important_features.index.values)

        assign_x_y_pipe_to_dict(vector_dict,
                                f'Important features {len(important_columns)} {important_columns}',
                                X_Data_one_hot_encoded[important_columns], Y_Data, logit_model)

    for key, item in vector_dict.items():
        print("\n===============  ", key, " ================\n")
        X_train, X_test, y_train, y_test = train_test_split(item[0], item[1], test_size=0.3, random_state=42)
        y_train = return_flattened_data(y_train)
        y_test = return_flattened_data(y_test)
        pipe = item[2]
        pipe.fit(X_train, y_train)
        y_pred, y_score = get_prediction_score(X_test, y_test, pipe)
        if not imbalance_detected:
            imbalanced_data(y_test, y_pred, y_score)
        else:
            balanced_data(y_test, y_pred, y_score)

        # plot hinge loss
        from sklearn.metrics import hinge_loss

        hinge_loss_value = hinge_loss(y_test, y_pred)
        print("Hinge loss: ", hinge_loss_value)
        print("Hinge loss is a loss function used in binary classification tasks. "
              "Hinge Loss is a loss function utilized within machine learning to train classifiers that optimize to "
              "increase the margin between data points and the decision boundary. Hence, it is mainly used for"
              " maximum margin classifications"
              ""
              "Commonly used in Support Vector Machines (SVMs), providing a margin-based approach to classification"
              "Alternative to Zero-one. For binary classification"
              ""
              "Maximize the margin- Hinge loss penalizes predictions that fall on the wrong side of the margin boundary,\n"
              "or are too close to the decision boundary, by  measuring how far data points are from the decision boundary, "
              "which helps approximate the likelihood of incorrect predictions. \n"
              "Hinge loss helps models generalize, making them more effective at accurately classifying data points. \n"
              "This helps ensure that the model can accurately classify data points with confidence.")
