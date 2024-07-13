from utils.visualization import plot_features, plot_histogram, plot_pairwise_features


def feature_analysis(D, L):
    plot_histogram(D, L)
    plot_features(D, L)
    plot_pairwise_features(D, L)
