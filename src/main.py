from data_preparer import DataPreparer
from k_means import K_Means

# Prepare the data before calculations
dp = DataPreparer(file_path='dataset/stck_data.csv')

# Create document-term matrix
dt_matrix = dp.apply_tf_idf()

# Run K-Means algorithm
k_means = K_Means(len(dp.classes), iterations=10, data=dt_matrix, data_length=dp.document_count)
clusters = k_means.cluster(dp.document_labels)
