import random
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


class K_Means:

    #
    def __init__(self, k: int, iterations: int, data: csr_matrix, data_length: int):
        self.k = k
        self.iterations = iterations
        self.data = data
        self.data_length = data_length

    #
    def __initialize_centroids(self):
        centroids = []
        centroid_norms = []
        for random_index in random.sample(range(0, self.data_length - 1), self.k):
            centroid = self.data.getrow(random_index).toarray()[0]
            centroids.append(centroid)

        return centroid_norms, centroids

    #
    def cluster(self, document_labels):
        centroid_norms, centroids = self.__initialize_centroids()

        # Main loop
        clusters = defaultdict(set)
        for i in range(self.iterations):
            clusters = defaultdict(set)  # Reset the clusters

            # find the distance between the point and cluster; choose the nearest centroid
            for row_index, sparse_row in enumerate(self.data):
                vector = sparse_row.toarray()

                closest_centroid = (float('inf'), None)
                for centroid_index, centroid in enumerate(centroids):
                    # Calculate inverse of cosine similarity
                    dist = 1 - np.true_divide(np.dot(vector, centroid), np.multiply(np.linalg.norm(vector), np.linalg.norm(centroid)))

                    if dist <= closest_centroid[0]:
                        closest_centroid = (dist, centroid_index)

                clusters[closest_centroid[1]].add(row_index)

            # Re-calculate centroids
            for centroid_index, vector_indices in clusters.items():
                avg_vector = None
                for vector_index in vector_indices:
                    if avg_vector is None:
                        avg_vector = self.data.getrow(vector_index).toarray()
                    else:
                        avg_vector = np.add(avg_vector, self.data.getrow(vector_index).toarray())

                centroids[centroid_index] = (avg_vector / len(vector_indices))[0]

            # Calculate purity
            majority_sum = 0
            for cluster in clusters.values():
                # Count cluster items with respect to their labels
                labeled_document_counts = defaultdict(int)
                for document_index in cluster:
                    labeled_document_counts[document_labels[document_index]] += 1

                # Find majority class
                majority_class = (0, None)
                for label, count in labeled_document_counts.items():
                    if count > majority_class[0]:
                        majority_class = (count, label)

                majority_sum += majority_class[0]  # Add majority to global sum

            print('Purity at iteration {} is\t{}'.format(i, (majority_sum / self.data_length)))

        return clusters
