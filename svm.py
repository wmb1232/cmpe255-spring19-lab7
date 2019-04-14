import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
# info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# load data
bankdata = pd.read_csv("./bill_authentication.csv")

# see the data
bankdata.shape

# see head
bankdata.head()

# data processing
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# train the SVM
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

# predictions
y_pred = svclassifier.predict(X_test)

# Evaluate model
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# TODO output

"""

"""

# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames)

    # process
    X = irisdata.drop('Class', axis=1)
    y = irisdata['Class']

    # train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # TODO: Evaluates perfomance of Polynomial Kernel, Gaussian Kernel, and Sigmoid Kernel.

    # train the SVM
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    #POLYNOMIAL KERNEL
    poly_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=8, coef0=100, C=5))
        ])

    poly_kernel_svm_clf.fit(X_train, y_train)

    # predictions
    y_pred = poly_kernel_svm_clf.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print('POLINOMIAL KERNEL')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # GAUSSIAN KERNEL
    poly_kernel_svm_clf_gaussian = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", degree=8, coef0=100, C=5))
    ])

    poly_kernel_svm_clf_gaussian.fit(X_train, y_train)

    # predictions
    y_pred = poly_kernel_svm_clf_gaussian.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print('GAUSSIAN KERNEL')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # SIGMOID KERNEL
    poly_kernel_svm_clf_sigmoid = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="sigmoid", degree=8, coef0=1, C=3))
    ])

    poly_kernel_svm_clf_sigmoid.fit(X_train, y_train)

    # predictions
    y_pred = poly_kernel_svm_clf_sigmoid.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print('SIGMOID KERNEL')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return [poly_kernel_svm_clf, poly_kernel_svm_clf_gaussian, poly_kernel_svm_clf_sigmoid]


models = import_iris()

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def display_svm_poly_kernel_iris():
    from sklearn.decomposition import PCA
    from sklearn import datasets


    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # # Read dataset to pandas dataframe
    # irisdata = pd.read_csv(url, names=colnames)
    #
    # # process
    # X = irisdata.drop('Class', axis=1)
    # y = irisdata['Class']
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    # train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    '''
    Polynomial Kernel using PCA data from iris
    '''
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=2, coef0=100, C=5))
    ])

    print('before model fit')
    poly_kernel_svm_clf.fit(X_train, y_train)
    print('after model fit')

    # predictions
    print('before model prediction')
    y_pred_clf = poly_kernel_svm_clf.predict(X_test)
    print('after model prediction')
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Polynomial kernel using PCA with 2 dimensions')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, poly_kernel_svm_clf, xx, yy, cmap=plt.cm.RdBu, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.RdBu, s=20, edgecolors='k')
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('Polynomial kernel using PCA with 2 dimensions')
    ax.legend()

    plt.show()

def display_svm_gaussian_kernel_iris():
    from sklearn.decomposition import PCA
    from sklearn import datasets


    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # # Read dataset to pandas dataframe
    # irisdata = pd.read_csv(url, names=colnames)
    #
    # # process
    # X = irisdata.drop('Class', axis=1)
    # y = irisdata['Class']
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    # train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    '''
    Gaussian Kernel using PCA data from iris
    '''
    # GAUSSIAN KERNEL
    poly_kernel_svm_clf_gaussian = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", degree=8, coef0=100, C=5))
    ])

    poly_kernel_svm_clf_gaussian.fit(X_train, y_train)

    # predictions
    print('before model prediction')
    y_pred_clf = poly_kernel_svm_clf_gaussian.predict(X_test)
    print('after model prediction')
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Gaussian kernel using PCA with 2 dimensions')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, poly_kernel_svm_clf_gaussian, xx, yy, cmap=plt.cm.BrBG, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.BrBG, s=20, edgecolors='k')
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('Gaussian kernel using PCA with 2 dimensions')
    ax.legend()

    plt.show()

def display_svm_sigmoid_kernel_iris():
    from sklearn.decomposition import PCA
    from sklearn import datasets


    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # # Read dataset to pandas dataframe
    # irisdata = pd.read_csv(url, names=colnames)
    #
    # # process
    # X = irisdata.drop('Class', axis=1)
    # y = irisdata['Class']
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    # train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    '''
    Sigmoid Kernel using PCA data from iris
    '''
    # SIGMOID KERNEL
    poly_kernel_svm_clf_sigmoid = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="sigmoid", degree=2, coef0=3, C=3))
    ])

    poly_kernel_svm_clf_sigmoid.fit(X_train, y_train)

    # predictions
    print('before model prediction')
    y_pred_clf = poly_kernel_svm_clf_sigmoid.predict(X_test)
    print('after model prediction')
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Sigmoid kernel using PCA with 2 dimensions')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, poly_kernel_svm_clf_sigmoid, xx, yy, cmap=plt.cm.Spectral, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.Spectral, s=20, edgecolors='k')
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('Sigmoid kernel using PCA with 2 dimensions')
    ax.legend()

    plt.show()


display_svm_poly_kernel_iris()
display_svm_gaussian_kernel_iris()
display_svm_sigmoid_kernel_iris()
