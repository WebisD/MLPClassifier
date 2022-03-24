from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_dataset():
    resultFromFile = load_iris()

    features = resultFromFile.data
    target = resultFromFile.target

    return resultFromFile, features, target


def plot_original_data(original_features, target, subplotPosition = 1):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, subplotPosition)
    plt.scatter(original_features[:, 0], original_features[:, 1], c=target, marker='o', cmap='viridis')


def plot_classifier(classifier, features, target, subplotPosition):
    classifier.fit(features, target)
    prediction = classifier.predict(features)

    plt.subplot(2, 2, subplotPosition)
    plt.scatter(features[:, 0], features[:, 1], c=prediction, marker='d', cmap='viridis', s=150)
    plt.scatter(features[:, 0], features[:, 1], c=target, marker='o', cmap='viridis', s=15)


def reduce_dimensions_with_cpa(original_features, target, desiredDimension, subplotPosition):
    pca = PCA(n_components=desiredDimension, whiten=True, svd_solver='randomized')
    pca = pca.fit(original_features)
    pca_features = pca.transform(original_features)
    print('Keep %5.2f%% of data from initial dataset'%(sum(pca.explained_variance_ratio_)*100))

    plt.subplot(2,2,subplotPosition)
    plt.scatter(pca_features[:,0], pca_features[:,1], c=target,marker='o',cmap='viridis')

    return pca_features


def plot_all_confusion_matrix(target, data, classifiers):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    for c, ax in zip(classifiers, axes.flatten()):
        title, classifier, features = c
        plot_confusion_matrix(classifier,
                              features,
                              target,
                              ax=ax,
                              display_labels=data.target_names)
        ax.title.set_text(title)

    plt.tight_layout()

    plt.show()

def main():
    data, original_features, target = load_dataset()
    plot_original_data(original_features, target)

    # without PCA
    original_classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 50, 160), activation="relu", alpha=1, max_iter=8000)
    plot_classifier(original_classifier, original_features, target, subplotPosition=2)

    # with PCA
    pca_features = reduce_dimensions_with_cpa(original_features, target, desiredDimension=2, subplotPosition=3)

    pca_classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 100, 100), alpha=1, max_iter=7500)
    plot_classifier(pca_classifier, pca_features, target, subplotPosition=4)

    # graphs plot
    plt.show()

    # confusion matrix plot
    plot_all_confusion_matrix(target, data, [
        ("Without PCA", original_classifier, original_features),
        ("With PCA", pca_classifier, pca_features)
    ])

main()
