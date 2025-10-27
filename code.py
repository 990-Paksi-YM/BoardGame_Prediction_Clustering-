import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/MyDrive/dbimbing/boardgame-geek-dataset_organized.csv')
data.sample (n=5)

data.info()

'''
We will not use the release_year, Amazon price, standard_deviation, comment, and columns 34-56 
because of the large amount of incomplete data that cannot be used and is irrelevant to the analysis being conducted.
'''
# Drop columns by index
columns_to_drop = [2, 17, 18, 19] + list(range(24, 57))
data2 = data.drop(data.columns[columns_to_drop], axis=1)

data2.info()

#EDA

# Select columns from index 2 to 29 from data2
columns_for_boxplots = data2.iloc[:, 2:].columns

# Create an interactive boxplot for each selected column
for col in columns_for_boxplots:
    fig = px.box(data2, y=col, title=f"Interactive Boxplot for {col}")
    fig.update_layout(yaxis_title=col)
    fig.show()

# a lot of data is outside the upper limit of the data (extreme data)

# Select columns from index 2 to 29 from data2
selected_columns = data2.iloc[:, 2:]

# Calculate the correlation matrix
correlation_matrix = selected_columns.corr()

# Create a heatmap
plt.figure(figsize=(18, 14)) # Adjust figure size as needed for readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix for Selected Columns')
plt.tight_layout()
plt.show()

#avg_rating has a moderate relation with complexity

# Select columns from index 2 to 29 from data2
selected_columns = data2.iloc[:, 2:]

# Create interactive histograms for each selected column
for col in selected_columns.columns:
    fig = px.histogram(data2, x=col, title=f"Interactive Histogram for {col}")
    fig.update_layout(xaxis_title=col, yaxis_title="Count")
    fig.show()


#Most data is more skewed to the left or skews


# Select columns from index 2 to 29 from data2
selected_columns = data2.iloc[:, 2:]

# Create static scatter plots for pairs of selected columns
for i in range(len(selected_columns.columns)):
    for j in range(i + 1, len(selected_columns.columns)):
        col1 = selected_columns.columns[i]
        col2 = selected_columns.columns[j]
        plt.figure(figsize=(8, 6))
        plt.scatter(selected_columns[col1], selected_columns[col2])
        plt.title(f"Scatter Plot: {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()

'''
The correlation between avg_rating and complexity appears to have a positive correlation, while avg_rating and rank_overall appear to have a negative correlation.
In the analysis of the average rating, we will use feature complexity and overall rank.
'''

data_train = data2.iloc[:, 2:]
data_test = data2.iloc[:, 2:]

import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

# Data dengan outlier
X = data_train

# Inisialisasi scaler
robust_scaler = RobustScaler()
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Transformasi data dan konversi kembali ke DataFrame
X_robust = pd.DataFrame(robust_scaler.fit_transform(X), columns=X.columns)
X_minmax = pd.DataFrame(minmax_scaler.fit_transform(X), columns=X.columns)
X_standard = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)

# Cetak hasil
print("Data asli:")
display(X.head()) # Display the head of the DataFrame

print("\nRobustScaler:")
display(X_robust.head()) # Display the first 5 rows of the DataFrame

print("\nMinMaxScaler:")
display(X_minmax.head()) # Display the first 5 rows of the DataFrame

print("\nStandardScaler:")
display(X_standard.head()) # Display the first 5 rows of the DataFrame


# Data dengan outlier
Y = data_test

# Inisialisasi scaler
robust_scaler = RobustScaler()
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Transformasi data dan konversi kembali ke DataFrame
Y_robust = pd.DataFrame(robust_scaler.fit_transform(Y), columns=Y.columns)
Y_minmax = pd.DataFrame(minmax_scaler.fit_transform(Y), columns=Y.columns)
Y_standard = pd.DataFrame(standard_scaler.fit_transform(Y), columns=Y.columns)

# Cetak hasil
print("Data asli:")
display(Y.head()) # Display the head of the DataFrame

print("\nRobustScaler:")
display(Y_robust.head()) # Display the first 5 rows of the DataFrame

print("\nMinMaxScaler:")
display(Y_minmax.head()) # Display the first 5 rows of the DataFrame

print("\nStandardScaler:")
display(Y_standard.head()) # Display the first 5 rows of the DataFrame

'''
for data normalization, will use the results of the minmaxscaler method,
avg_rating Prediction using Regression Linear
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target variable from the normalized training data (X_minmax)
X_train = X_minmax[['complexity', 'rank_overall']]
y_train = X_minmax['avg_rating']

# Select features and target variable from the normalized testing data (Y_minmax)
X_test = Y_minmax[['complexity', 'rank_overall']]
y_test = Y_minmax['avg_rating']

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Print the coefficient and intercept
print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Display the first few predictions and actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
display(results.head())

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Avg Rating")
plt.ylabel("Predicted Avg Rating")
plt.title("Actual vs Predicted Avg Rating")
plt.show()



# Select features and target variable from the normalized training data (X_minmax)
X_train = X_minmax.drop('avg_rating', axis=1)
y_train = X_minmax['avg_rating']

# Select features and target variable from the normalized testing data (Y_minmax)
X_test = Y_minmax.drop('avg_rating', axis=1)
y_test = Y_minmax['avg_rating']


# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Print the coefficient and intercept
print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Display the first few predictions and actual values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
display(results.head())

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Avg Rating")
plt.ylabel("Predicted Avg Rating")
plt.title("Actual vs Predicted Avg Rating")
plt.show()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Select features (X) and target variable (y) from the normalized data
X = X_minmax.drop('avg_rating', axis=1) # Using all columns except 'avg_rating'
y = X_minmax['avg_rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42) # You can adjust n_estimators
model_rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = model_rf.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regressor (with train_test_split) - Mean Squared Error: {mse_rf}")
print(f"Random Forest Regressor (with train_test_split) - R-squared: {r2_rf}")

# Display the first few predictions and actual values
results_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
display(results_rf.head())

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Avg Rating")
plt.ylabel("Predicted Avg Rating")
plt.title("Random Forest Regressor (with train_test_split) - Actual vs Predicted Avg Rating")
plt.show()

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Select the data for clustering (excluding the target variable if it's not needed for clustering)
# Based on previous analysis, 'avg_rating' was the target for regression, so we'll exclude it for clustering.
X_clustering = X_minmax.drop('avg_rating', axis=1)

# Define the range of clusters to try
n_clusters_range = [2, 3, 4, 5]

# Perform K-Means clustering for each number of clusters and visualize
for n_clusters in n_clusters_range:
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init for clarity
    kmeans.fit(X_clustering)

    # Add the cluster labels to the original data for analysis
    data2[f'cluster_kmeans_{n_clusters}'] = kmeans.labels_

    print(f"Cluster centers for {n_clusters} clusters:")
    display(pd.DataFrame(kmeans.cluster_centers_, columns=X_clustering.columns))

    # --- Visualization ---
    print(f"Visualizing clusters for {n_clusters} clusters...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_clustering)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data2[f'cluster_kmeans_{n_clusters}'], palette='viridis', legend='full')
    plt.title(f'K-Means Clustering with {n_clusters} Clusters (PCA 2D)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    # --- End Visualization ---


print("\nClustering complete. Cluster labels have been added to the 'data2' DataFrame.")
display(data2.head())

# Select the data for clustering (using Y_minmax as requested)
X_clustering = Y_minmax.drop('avg_rating', axis=1)

# Calculate inertia for a range of cluster numbers
inertia = []
k_range = range(1, 11) # You can adjust the range as needed

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_clustering)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (using Y_minmax)')
plt.xticks(k_range)
plt.grid(True)

# Add labels to the points
for k, inert in zip(k_range, inertia):
    plt.text(k, inert, f'{inert:.2f}', ha='right', va='bottom')

plt.show()

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Select the data for clustering (excluding the target variable if it's not needed for clustering)
X_clustering = Y_minmax.drop('avg_rating', axis=1)

# Define the range of clusters to evaluate
n_clusters_range = [2, 3, 4, 5]

# Calculate and print evaluation metrics for each number of clusters
silhouette_scores = []
dbi_scores = []
ch_scores = []

print("Evaluating clustering for different numbers of clusters (using Y_minmax):")
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_clustering)
    cluster_labels = kmeans.labels_

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(X_clustering, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    # Calculate Davies-Bouldin Index
    dbi = davies_bouldin_score(X_clustering, cluster_labels)
    dbi_scores.append(dbi)

    # Calculate Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(X_clustering, cluster_labels)
    ch_scores.append(ch_score)

    print(f"\nNumber of Clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Davies-Bouldin Index: {dbi:.4f}")
    print(f"  Calinski-Harabasz Index: {ch_score:.4f}")

# Optional: Plot the evaluation scores (you can choose to plot them separately or together)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score (using Y_minmax)')
plt.xticks(n_clusters_range)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(n_clusters_range, dbi_scores, marker='o', color='orange')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index (using Y_minmax)')
plt.xticks(n_clusters_range)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(n_clusters_range, ch_scores, marker='o', color='green')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Index')
plt.title('Calinski-Harabasz Index (using Y_minmax)')
plt.xticks(n_clusters_range)
plt.grid(True)

plt.tight_layout()
plt.show()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Select the data for clustering (using Y_minmax for evaluation as requested)
X_clustering = Y_minmax.drop('avg_rating', axis=1)

# Define the range of clusters to evaluate
n_clusters_range = [2, 3, 4, 5] # Evaluate for the same range as K-Means

print("Evaluating Hierarchical Clustering for different numbers of clusters (using Y_minmax):")

# Prepare PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)

for n_clusters in n_clusters_range:
    print(f"\nNumber of Clusters: {n_clusters}")

    # Perform Hierarchical Clustering
    hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = hierarchical_clustering.fit_predict(X_clustering)

    # Calculate evaluation metrics
    try:
        silhouette_avg = silhouette_score(X_clustering, cluster_labels)
        print(f"  Silhouette Score: {silhouette_avg:.4f}")
    except Exception as e:
        print(f"  Could not calculate Silhouette Score: {e}")

    try:
        dbi = davies_bouldin_score(X_clustering, cluster_labels)
        print(f"  Davies-Bouldin Index: {dbi:.4f}")
    except Exception as e:
        print(f"  Could not calculate Davies-Bouldin Index: {e}")

    try:
        ch_score = calinski_harabasz_score(X_clustering, cluster_labels)
        print(f"  Calinski-Harabasz Index: {ch_score:.4f}")
    except Exception as e:
        print(f"  Could not calculate Calinski-Harabasz Index: {e}")

    # --- Visualization ---
    print(f"Visualizing clusters for {n_clusters} clusters...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', legend='full')
    plt.title(f'Hierarchical Clustering with {n_clusters} Clusters (PCA 2D, using Y_minmax)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    # --- End Visualization ---

print("\nHierarchical Clustering evaluation complete (using Y_minmax).")


# Implement Linear Regression prediction on the actual data (data2)

# Select the features for prediction from data2
X_actual = data2[['complexity', 'rank_overall']]

# Scale the actual features using a scaler fitted on the original range of the training features
from sklearn.preprocessing import MinMaxScaler
scaler_features = MinMaxScaler()

# Fit the scaler ONLY on the original features that correspond to the training features ('complexity' and 'rank_overall') from data2.iloc[:, 2:]
original_training_features = data2.iloc[:, 2:][['complexity', 'rank_overall']]
scaler_features.fit(original_training_features)

# Transform the actual data from data2 using the fitted features scaler
X_actual_scaled = scaler_features.transform(X_actual)

# Re-train the Linear Regression model on the correct features from X_minmax
from sklearn.linear_model import LinearRegression
model_for_prediction = LinearRegression()
X_train_model = X_minmax[['complexity', 'rank_overall']]
y_train_model = X_minmax['avg_rating']
model_for_prediction.fit(X_train_model, y_train_model)

# Make predictions in the scaled range
predicted_avg_rating_scaled = model_for_prediction.predict(X_actual_scaled)

# Inverse transform the predictions back to the original scale of 'avg_rating'
scaler_avg_rating = MinMaxScaler()
# Fit the scaler for avg_rating on the original 'avg_rating' values from the training data
original_avg_rating = data2.iloc[:, 2:]['avg_rating'].values.reshape(-1, 1) # Reshape for scaler
scaler_avg_rating.fit(original_avg_rating)

# Inverse transform the scaled predictions
data2['predicted_avg_rating_lr'] = scaler_avg_rating.inverse_transform(predicted_avg_rating_scaled.reshape(-1, 1)) # Reshape for inverse_transform


# Display data2 with the new prediction column
print("Data2 with Linear Regression predictions:")
display(data2[['boardgame', 'avg_rating', 'complexity', 'rank_overall', 'predicted_avg_rating_lr']].head())


# Implement Random Forest prediction on the actual data (data2)

# Select the features for prediction from data2 that match the original training features (17 columns)
# These are the columns in data2.iloc[:, 2:] excluding 'avg_rating', BEFORE cluster/prediction columns were added.
# We need to get the list of original feature columns.
original_feature_columns = X_minmax.drop('avg_rating', axis=1).columns.tolist()

X_actual_rf = data2[original_feature_columns]


# Scale the actual features using the same scaler fitted on the original training features
from sklearn.preprocessing import MinMaxScaler

# Assuming the scaler used for X_minmax was fitted on data2.iloc[:, 2:].drop('avg_rating', axis=1)
# We need to re-create and fit the scaler on the original training features
scaler_features_rf = MinMaxScaler()
X_train_features_for_scaler_rf = data2[original_feature_columns] # Fit on original data range
scaler_features_rf.fit(X_train_features_for_scaler_rf)

# Transform the actual data from data2 using the fitted features scaler
X_actual_scaled_rf = scaler_features_rf.transform(X_actual_rf)

# Use the trained Random Forest Regressor model to make predictions
# Assuming the trained model from cell fo_Jq-lpynxy is available as 'model_rf'

# Make predictions in the scaled range
predicted_avg_rating_scaled_rf = model_rf.predict(X_actual_scaled_rf)

# Inverse transform the predictions back to the original scale of 'avg_rating'
# Assuming the scaler for avg_rating was fitted on the original 'avg_rating' values from the training data
# We can reuse the scaler_avg_rating fitted in the Linear Regression implementation cell (XGUlwOQfo62S)
# If that cell hasn't been run, we need to fit it here
try:
    scaler_avg_rating # Check if scaler_avg_rating exists
except NameError:
    scaler_avg_rating = MinMaxScaler()
    original_avg_rating = data2.iloc[:, 2:]['avg_rating'].values.reshape(-1, 1) # Fit on original training avg_rating
    scaler_avg_rating.fit(original_avg_rating)


# Inverse transform the scaled predictions
data2['predicted_avg_rating_rf'] = scaler_avg_rating.inverse_transform(predicted_avg_rating_scaled_rf.reshape(-1, 1)) # Reshape for inverse_transform

# Display data2 with the new prediction column
print("Data2 with Random Forest predictions:")
display(data2[['boardgame', 'avg_rating', 'predicted_avg_rating_lr', 'predicted_avg_rating_rf']].head())



from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Select the features for clustering from data2 (using the original 17 features)
# We need to get the list of original feature columns again.
original_feature_columns = X_minmax.drop('avg_rating', axis=1).columns.tolist()
X_actual_clustering = data2[original_feature_columns]

# Scale the actual features using the same scaler fitted on the original training features
scaler_clustering_features = MinMaxScaler()
# Fit the scaler on the original training features (the same set of columns from data2)
X_train_features_for_scaler_clustering = data2[original_feature_columns]
scaler_clustering_features.fit(X_train_features_for_scaler_clustering)

# Transform the actual data from data2 using the fitted features scaler
X_actual_scaled_clustering = scaler_clustering_features.transform(X_actual_clustering)

# Perform K-Means clustering with 3 clusters
n_clusters_kmeans = 3
kmeans_actual = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
cluster_labels_actual = kmeans_actual.fit_predict(X_actual_scaled_clustering)

# Add the cluster labels to the original data2 DataFrame
data2[f'cluster_kmeans_{n_clusters_kmeans}_actual'] = cluster_labels_actual

# Display the entire data2 DataFrame with all columns and the new cluster column
print(f"Data2 with K-Means ({n_clusters_kmeans} clusters) labels:")
display(data2)

# Optional: Display cluster centers on scaled data
print(f"\nCluster centers for K-Means ({n_clusters_kmeans} clusters) on scaled data:")
display(pd.DataFrame(kmeans_actual.cluster_centers_, columns=original_feature_columns))



from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

# Select the features for clustering from data2 (using the original 17 features)
# We need to get the list of original feature columns again.
# Assuming original_feature_columns was defined in a previous cell (e.g., hc03mTJ91iwC or N-j5qEUfdiL5)
try:
    original_feature_columns
except NameError:
    # If not defined, re-create it by dropping 'avg_rating' from the columns of X_minmax
    original_feature_columns = X_minmax.drop('avg_rating', axis=1).columns.tolist()


X_actual_clustering_hierarchical = data2[original_feature_columns]

# Scale the actual features using the same scaler fitted on the original training features
# Assuming scaler_clustering_features was fitted in the K-Means implementation cell (N-j5qEUfdiL5)
try:
    scaler_clustering_features
except NameError:
     # If not defined, re-create and fit it on the original training features
    scaler_clustering_features = MinMaxScaler()
    X_train_features_for_scaler_clustering = data2[original_feature_columns]
    scaler_clustering_features.fit(X_train_features_for_scaler_clustering)

# Transform the actual data from data2 using the fitted features scaler
X_actual_scaled_clustering_hierarchical = scaler_clustering_features.transform(X_actual_clustering_hierarchical)

# Perform Hierarchical Clustering
n_clusters_hierarchical = 3 # You can change this number
hierarchical_clustering_actual = AgglomerativeClustering(n_clusters=n_clusters_hierarchical)
cluster_labels_hierarchical_actual = hierarchical_clustering_actual.fit_predict(X_actual_scaled_clustering_hierarchical)

# Add the cluster labels to the original data2 DataFrame
data2[f'cluster_hierarchical_{n_clusters_hierarchical}_actual'] = cluster_labels_hierarchical_actual

# Display data2 with the new cluster column
print(f"Data2 with Hierarchical Clustering ({n_clusters_hierarchical} clusters) labels:")
display(data2)
