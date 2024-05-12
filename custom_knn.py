from math import sqrt

class KNN():
    def __init__(self, k):
        self.k = k
        print(self.k)
        
    def fit(self, X_train, y_train):
        self.x_train = X_train
        self.y_train = y_train

    def calculate_euclidean(self, sample1, sample2):
        distance = 0.0
        for i in range(len(sample1)):
            distance += (sample1[i] - sample2[i])**2
        return sqrt(distance)
    
    def nearest_neighbors(self, test_sample):
        distances = []
        for i in range(len(self.x_train)):
            distances.append((self.y_train[i], self.calculate_euclidean(self.x_train[i], test_sample), i))
        distances.sort(key=lambda x: x[1])
        neighbors = []
        for i in range(self.k):
            neighbors.append((distances[i][0], distances[i][2]))
        return neighbors

    def predict(self, test_set):
        predictions = []
        for test_sample in test_set:
            neighbors = self.nearest_neighbors(test_sample)
            labels = [sample for sample in neighbors]
            prediction = max(labels, key=labels.count)
            predictions.append(prediction)
        return predictions
    
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        correct = sum(1 for pred, actual in zip(predictions, y_test) if pred == actual)
        return correct / len(y_test)
    
#     # def predict(self, test_sample):
#     #     neighbors = self.nearest_neighbors(test_sample)
#     #     rating_counts = {}
#     #     for rating, _ in neighbors:
#     #         if rating in rating_counts:
#     #             rating_counts[rating] += 1
#     #         else:
#     #             rating_counts[rating] = 1
#     #     predicted_rating = max(rating_counts, key=rating_counts.get)  # Get key with highest count
#     #     return predicted_rating

#     # def calculate_accuracy(self, predictions, true_labels):
#     #     correct = 0
#     #     for i in range(len(predictions)):
#     #         if predictions[i] == true_labels[i]:
#     #             correct += 1
#     #     accuracy = correct / len(predictions)
#     #     return accuracy

# from math import sqrt
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.model_selection import cross_val_score

# class KNN(BaseEstimator, ClassifierMixin):
#     def __init__(self, k):
#         self.k = k
        
#     def fit(self, X_train, y_train):
#         self.x_train = X_train
#         self.y_train = y_train

#     def calculate_euclidean(self, sample1, sample2):
#         distance = 0.0
#         for i in range(len(sample1)):
#             distance += (sample1[i] - sample2[i])**2
#         return sqrt(distance)
    
#     def nearest_neighbors(self, test_sample):
#         distances = []
#         for i in range(len(self.x_train)):
#             distances.append((self.y_train[i], self.calculate_euclidean(self.x_train[i], test_sample), i))
#         distances.sort(key=lambda x: x[1])
#         neighbors = []
#         for i in range(self.k):
#             neighbors.append((distances[i][0], distances[i][2]))
#         return neighbors

#     def predict(self, test_set):
#         predictions = []
#         for test_sample in test_set:
#             neighbors = self.nearest_neighbors(test_sample)
#             labels = [sample[0] for sample in neighbors]  # Extract labels from neighbors
#             prediction = max(labels, key=labels.count)
#             predictions.append(prediction)
#         return predictions

#     def score(self, X, y):
#         predictions = self.predict(X)
#         correct = sum(1 for pred, actual in zip(predictions, y) if pred == actual)
#         return correct / len(y)
