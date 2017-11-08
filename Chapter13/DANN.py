#https://github.com/alwc/ML_algorithms
# -*- coding: utf-8 -*-
class DANN(object):
    """
    Discriminant Adaptive Nearest Neighbors (DANN).
    DANN adaptively elongates neighborhoods along boundry regions.
    Useful for high dimensional data.
    Reference:
        Hastie, Trevor, and Robert Tibshirani.
        "Discriminant adaptive nearest neighbor classification."
        IEEE transactions on pattern analysis and machine intelligence
        18.6 (1996): 607-616.
    """
    def __init__(self):
        """
        Attributes:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            neighborhood_size (int): number of nearest neighbors to consider when predicting
            learned (bool): Keeps track of if model has been fit
        """
        self.X = None
        self.y = None
        self.neighborhood_size = None
        self.learned = False

 def predict(self, x, k=10):
        """
        Args:
            x1 (np.array): query point of shape[n_features]
            k (int): number of nearest neighbors to consider
        Returns:
            prediction: Returns predicted class of sample
        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        n_features = len(x)
        distances = []
        for row in self.X:
            distance = np.linalg.norm(row-x)
            distances.append(distance)
        distances = np.array(distances)
        nearest_neighbors = np.argsort(distances)[:self.neighborhood_size]
        neighborhood_X = self.X[nearest_neighbors, :]
        neighborhood_X_mean = neighborhood_X.mean(axis=0)
        neighborhood_y = self.y[nearest_neighbors]
        neighborhood_classes = np.unique(neighborhood_y)
        class_frequencies = {}
        within_class_cov = np.zeros((n_features, n_features))
        between_class_cov = np.zeros((n_features, n_features))
        for target_class in neighborhood_classes:
            class_indices = np.where(neighborhood_y == target_class)[0]
            class_frequencies[target_class] = np.sum(neighborhood_y == target_class) / self.neighborhood_size
            class_covariance = np.cov(neighborhood_X[class_indices, :], rowvar=False)
            within_class_cov += class_covariance * class_frequencies[target_class]
            class_mean = neighborhood_X[class_indices, :].mean(axis=0)
            between_class_cov += np.outer(class_mean - neighborhood_X_mean, class_mean - neighborhood_X_mean) * class_frequencies[target_class]
        # W* = W^-.5
        # B* = W*BW*
        W_star = np.linalg.pinv(np.nan_to_num(np.power(within_class_cov, 0.5)))
        B_star = np.dot(W_star, between_class_cov).dot(W_star)
        I = np.identity(n_features)
        sigma = W_star.dot(B_star + self.epsilon * I).dot(W_star)
        distances = []
        for row in self.X:
            distances.append(self.DANN_distance(x, row, sigma))
        distances = np.array(distances)
        nearest = distances.argsort()[:k]
        prediction = stats.mode(self.y[nearest]).mode[0]
        return prediction

        def DANN_distance(self, x0, x1, sigma):
            """
            Computes the distance between x0 and x1 using the DANN metric
            which is adaptively defined at query locus
            Args:
                x1 (np.array): query point of shape[n_features]
                x2 (np.array): reference point of shape[n_features]
                sigma (np.ndarray): array of shape[n_features, n_features]
            """
            difference = x0 - x1
            distance = difference.T.dot(sigma).dot(difference)
            return distance
