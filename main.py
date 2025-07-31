from datetime import datetime
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('/content/bank_customer_transactions (1).csv')
print(df.head())

# Data Preprocessing
ast_7_cols = df.columns[-7:]
# Merge them row-wise into a single string, skipping NaNs
df['OwnedProducts'] = df[last_7_cols].astype(str).apply(
    lambda row: ' '.join([val for val in row if val != 'nan']), axis=1
)
# Drop the original 7 columns
last_6_cols = df.columns[-6:]
df.drop(columns=last_6_cols, inplace=True)


#shape of data
print(df.shape)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("*" * 50)
# Basic Info
print(df.info())
print("*" * 50)
# Summary stats
print(df.describe())
print("*" * 50)
# Missing values
print(df.isnull().sum())
print("*" * 50)
# Duplicates
print(f"Duplicated rows: {df.duplicated().sum()}")


def clean_owned_products(entry):
    if pd.isnull(entry):
        return ""
    entry = entry.replace("+ACI-", "").replace("#NAME?", "").replace("+AC0-", "-").strip()
    products = re.split(r'\s{2,}|\t|\s{1}', entry)
    cleaned = [p.strip() for p in products if p.strip()]
    return ", ".join(cleaned)
df['OwnedProducts'] = df['OwnedProducts'].apply(clean_owned_products)
df['CustomerDOB'] = df['CustomerDOB'].str.replace(r'\+AC0-', '-', regex=True)


# Ensure CustomerDOB is in datetime format
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], errors='coerce')
# Calculate age
today = datetime.today()
df['Age'] = df['CustomerDOB'].apply(
    lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if pd.notnull(dob) else None
)
# Adjust age > 200
df.loc[df['Age'] > 200, 'Age'] = df['Age'] - 200
# Take absolute value
df['Age'] = df['Age'].abs()
# Drop rows where Age is less than 21
df = df[df['Age'] >= 21]
print(df["Age"].value_counts())


# Create a histogram of the 'Age' column
plt.hist(df['Age'], bins=20, edgecolor='black')  # Adjust bins as needed
# Customize the plot (add labels, title, etc.)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Customer Ages')
# Display the plot
plt.show()

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], dayfirst=True, errors='coerce')
pd.set_option('display.max_rows', None)  # Remove row limit for output
df['CustLocation'].value_counts()
#Remove extra spaces and symbols:
df['CustLocation'] = df['CustLocation'].str.replace(r'[^A-Z\s]', '', regex=True)
df['CustLocation'] = df['CustLocation'].str.replace(r'\s+', ' ', regex=True).str.strip()


# Step 1: Get top N real locations (manually verified as clean)
top_locations = df['CustLocation'].str.upper().value_counts().head(200).index.tolist()

# Step 2: Define a fuzzy matching function with caching
fuzzy_cache = {}

def fuzzy_match_location(loc, choices, threshold=50):
    if pd.isnull(loc) or loc.strip() == "":  # Handle missing or empty values
        return 'Unknown'
    loc = loc.upper().strip()  # Ensure uppercase and remove leading/trailing whitespace
    if loc in fuzzy_cache:  # Check cache
        return fuzzy_cache[loc]
    match, score = process.extractOne(loc, choices)
    result = match if score >= threshold else loc
    fuzzy_cache[loc] = result  # Cache the result
    return result

# Step 3: Preprocess CustLocation column to remove invalid entries
df['CustLocation'] = df['CustLocation'].astype(str).str.strip()  # Ensure all entries are strings and strip whitespace

# Step 4: Apply fuzzy matching to the CustLocation column
df['CustLocation'] = df['CustLocation'].apply(lambda x: fuzzy_match_location(x, top_locations))


df['CustLocation'].value_counts()
# Get the top N locations for better visualization (adjust N as needed)
top_n = 20
location_counts = df['CustLocation'].value_counts().head(top_n)

# Create the bar plot using seaborn
plt.figure(figsize=(12, 6))  # Adjust figure size as needed
sns.barplot(x=location_counts.index, y=location_counts.values)
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.xlabel('Customer Location')
plt.ylabel('Number of Customers')
plt.title(f'Top {top_n} Customer Locations')
plt.tight_layout()
plt.show()


missing_values = df.isna().sum()
print("Missing values per column:\n", missing_values)

# Fix: Convert CustAccountBalance to numeric (if needed)
df['CustAccountBalance'] = pd.to_numeric(df['CustAccountBalance'], errors='coerce')
# Impute CustAccountBalance with median
#Calculates the median CustAccountBalance for each Age group
df['CustAccountBalance'] = df.groupby('Age')['CustAccountBalance'].transform(
    lambda x: x.fillna(x.median())
)
# Drop rows with missing TransactionDate
df = df[df['TransactionDate'].notnull()]
df = df[df['CustGender'].notnull()]
# Show missing values per column
missing_values = df.isnull().sum()
print("Missing values per column (after cleanup):")
print(missing_values)

# Remove duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
df.drop_duplicates(inplace=True)

# Display unique values in the Gender column
print(df['CustGender'].unique())  

# Group by CustomerID and check unique gender values
conflicting_genders = df.groupby('CustomerID')['CustGender'].nunique()

# Identify customers with conflicting genders
conflicting_customers = conflicting_genders[conflicting_genders > 1].index
print(f"Customers with conflicting genders: {len(conflicting_customers)}")

# Display conflicting rows for inspection
conflicting_rows = df[df['CustomerID'].isin(conflicting_customers)]
conflicting_rows

# Step 1: Clean the CustGender column
df['CustGender'] = df['CustGender'].astype(str).str.strip().str.upper()

# Replace invalid gender entries (e.g., T, empty, etc.) with NaN
df.loc[~df['CustGender'].isin(['M', 'F']), 'CustGender'] = pd.NA

# Step 2: Standardize gender using the first valid gender per CustomerID
# Drop rows with missing CustGender temporarily to avoid mapping nulls
first_gender = df.dropna(subset=['CustGender']).groupby('CustomerID')['CustGender'].first()

# Map the first valid gender to all entries
df['CustGender'] = df['CustomerID'].map(first_gender)

# Step 3: Check for remaining conflicting genders per CustomerID
conflicting_customers = df.groupby('CustomerID')['CustGender'].nunique()
conflicting_customers = conflicting_customers[conflicting_customers > 1]

print("Gender has been standardized based on the first valid entry per customer.")
print(f"Remaining conflicting customers: {len(conflicting_customers)}")

# Drop rows where CustGender is 'T' directly
df.drop(df[df['CustGender'] == 'T'].index, inplace=True)
print(df['CustGender'].value_counts())

# Check how many are missing or empty strings
print("Missing (NaN):", df['OwnedProducts'].isna().sum())
print("Empty strings:", (df['OwnedProducts'].astype(str).str.strip() == '').sum())


df['OwnedProducts'] = df['OwnedProducts'].replace('', pd.NA)  # Convert empty strings to NaN
df['OwnedProducts'].fillna('Unknown', inplace=True)

df = df[df['OwnedProducts'].notna() & (df['OwnedProducts'].astype(str).str.strip() != '')]

#check after cleaning
print("Missing After Cleaning(NaN):", df['OwnedProducts'].isna().sum())
print("Empty strings After Cleaning:", (df['OwnedProducts'].astype(str).str.strip() == '').sum())

# Ensure 'CustomerID' exists in the dataset
if 'CustomerID' in df.columns:
    # Aggregate data for each customer
    customer_aggregated = df.groupby('CustomerID').agg({
        'TransactionAmount (INR)': 'sum',  # Total transaction amount
        'TransactionDate': 'count',       # Total number of transactions
        'CustAccountBalance': 'mean',     # Average account balance
        'Age': 'mean',                    # Average age
        'OwnedProducts': lambda x: ', '.join(x.unique())  # Combine unique owned products
    }).reset_index()

    # Rename columns for clarity
    customer_aggregated.rename(columns={
        'TransactionAmount (INR)': 'TotalTransactionAmount',
        'TransactionDate': 'TotalTransactions',
        'CustAccountBalance': 'AvgAccountBalance',
        'Age': 'AvgAge',
        'OwnedProducts': 'OwnedProductsList'
    }, inplace=True)

    # Display the aggregated DataFrame
    customer_aggregated.head()
else:
    print("The 'CustomerID' column is missing from the dataset.")


# Step 1: Split the products and create a list of all unique products
all_products = []
for products in customer_aggregated['OwnedProductsList']:
    for product in products.split(', '):
        if product not in all_products:  # Avoid duplicates
            all_products.append(product)

# Step 2: Create a new DataFrame with columns for each unique product
products_df = pd.DataFrame(index=customer_aggregated.index, columns=all_products).fillna(0)

# Step 3: Populate the DataFrame based on owned products
for index, row in customer_aggregated.iterrows():
    owned_products = row['OwnedProductsList'].split(', ')
    for product in owned_products:
        if product in all_products:
            products_df.loc[index, product] = 1

# Step 4: Concatenate the encoded features with the original DataFrame
customer_aggregated_encoded = pd.concat([customer_aggregated, products_df], axis=1)


# Select numerical features for scaling (excluding 'CustomerID' and one-hot encoded columns)
numerical_features = ['TotalTransactionAmount', 'TotalTransactions', 'AvgAccountBalance', 'AvgAge']
data_to_scale = customer_aggregated_encoded[numerical_features]

# Initialize and apply StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)

# Create a DataFrame with the scaled features
scaled_df = pd.DataFrame(scaled_data, columns=numerical_features, index=customer_aggregated_encoded.index)

# Concatenate scaled features with the rest of the data
final_df = pd.concat([customer_aggregated_encoded[['CustomerID', 'OwnedProductsList']],
                      scaled_df,
                      customer_aggregated_encoded.drop(columns=numerical_features + ['CustomerID', 'OwnedProductsList'])], axis=1)

# Display the final DataFrame
final_df.head()

# Test different PCA component numbers and plot explained variance
pca_features = final_df.drop(columns=['CustomerID', 'OwnedProductsList']).columns
scaled_features = final_df[pca_features]

variance_ratios = []
for n_components in range(1, len(pca_features) + 1):
    pca = PCA(n_components=n_components)
    pca.fit(scaled_features)
    variance_ratios.append(np.sum(pca.explained_variance_ratio_))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_features) + 1), variance_ratios, marker='o')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()

# Apply PCA
# Choose the number of components you want to keep
n_components = 5
pca = PCA(n_components=n_components)

# Fit PCA to the scaled data (excluding 'CustomerID' and 'OwnedProductsList')
pca_features = final_df.drop(columns=['CustomerID', 'OwnedProductsList']).columns
scaled_features = final_df[pca_features]  # Select the scaled features

principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)],
                      index=final_df.index)

# Concatenate PCA components with the rest of the data
final_df_pca = pd.concat([final_df[['CustomerID', 'OwnedProductsList']], pca_df], axis=1)

# Display the final DataFrame with PCA components
final_df_pca.head()


# Elbow method and Silhouette Score for optimal number of clusters
inertia_values = []
silhouette_scores = []

for n_clusters in range(2, 11):  # Test clusters from 2 to 10
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=32)
    kmeans.fit(final_df_pca.drop(columns=['CustomerID', 'OwnedProductsList']))

    inertia_values.append(kmeans.inertia_)

    # Calculate Silhouette Score on a sample
    sample_size = 10000  # Adjust sample size as needed
    sample_indices = np.random.choice(len(final_df_pca), size=sample_size, replace=False)
    sample_data = final_df_pca.drop(columns=['CustomerID', 'OwnedProductsList']).iloc[sample_indices]
    sample_clusters = kmeans.labels_[sample_indices]

    silhouette_avg = silhouette_score(sample_data, sample_clusters)
    silhouette_scores.append(silhouette_avg)

# Plot the inertia values (elbow method)
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Plot the Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# Elbow method and Silhouette Score for optimal number of clusters
inertia_values = []
silhouette_scores = []

for n_clusters in range(2, 11):  # Test clusters from 2 to 10
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=64)
    kmeans.fit(final_df_pca.drop(columns=['CustomerID', 'OwnedProductsList']))

    inertia_values.append(kmeans.inertia_)

    # Calculate Silhouette Score on a sample
    sample_size = 10000  # Adjust sample size as needed
    sample_indices = np.random.choice(len(final_df_pca), size=sample_size, replace=False)
    sample_data = final_df_pca.drop(columns=['CustomerID', 'OwnedProductsList']).iloc[sample_indices]
    sample_clusters = kmeans.labels_[sample_indices]

    silhouette_avg = silhouette_score(sample_data, sample_clusters)
    silhouette_scores.append(silhouette_avg)

# Plot the inertia values (elbow method)
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Plot the Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


# Assuming 'final_df_pca' is your DataFrame with PCA components
filtered_pca = final_df_pca.drop(columns=['CustomerID', 'OwnedProductsList'])

# Create a smaller sample for DBSCAN
pca_sample, _ = train_test_split(filtered_pca, test_size=0.9, random_state=42)

# Apply DBSCAN to the sample
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters if needed
dbscan_labels = dbscan.fit_predict(pca_sample)

# Add DBSCAN cluster labels to the original DataFrame (final_df_pca)
final_df_pca['DBSCAN_Cluster'] = np.nan  # Initialize a new column for DBSCAN labels
final_df_pca.loc[pca_sample.index, 'DBSCAN_Cluster'] = dbscan_labels

# Assuming 'final_df_pca' contains your data with DBSCAN cluster labels
plt.figure(figsize=(8, 6))  # Adjust figure size if needed

# Scatter plot of PC1 vs PC2, colored by DBSCAN cluster labels
plt.scatter(final_df_pca['PC1'], final_df_pca['PC2'], c=final_df_pca['DBSCAN_Cluster'], cmap='viridis')

# Customize the plot
plt.title('DBSCAN Clustering on PCA Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='DBSCAN Cluster')
plt.show()  # Display the plot
