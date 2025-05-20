# Re-import required libraries and reload the mental health dataset after kernel reset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.cluster.hierarchy import linkage, dendrogram
from PIL import Image
from IPython.display import display

# Load the dataset
df = pd.read_csv('mental_health_dataset.csv')
print("before")
print(df.head())
#Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
print("categorical variables encoded")
print(label_encoders)
print("after")
print(df.head())



# Standardize numerical features
scaler = StandardScaler()
features_to_scale = ['age', 'stress_level', 'sleep_hours', 'physical_activity_days', 'depression_score',
                     'anxiety_score', 'social_support_score', 'productivity_score']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("numerical features standardized")

print("Data visualization")
# # Display the first few rows of the dataset
display(df.head())



# Identify outliers and skewness before capping
outlier_counts = {}
skewness_values = {}

for col in features_to_scale:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_counts[col] = len(outliers)
    skewness_values[col] = df[col].skew()

# Combine results into a DataFrame
outlier_skew_df = pd.DataFrame({
    'Outlier Count': outlier_counts,
    'Skewness': skewness_values
}).sort_values(by='Outlier Count', ascending=False)

# Display the outlier and skewness DataFrame
print("Outlier and skewness DataFrame:")
display(outlier_skew_df)


# Generate visualizations for visualizing data and correlation analysis

# 1. Boxplot for all numerical features
plt.figure(figsize=(15, 6))
sns.boxplot(data=df[features_to_scale], orient="h", palette="Set2")
plt.title("Boxplot of All Numerical Features")
plt.xlabel("Standardized Values (Z-score)")
plt.tight_layout()
plt.show()

# 2. Histogram for all numerical features
plt.figure(figsize=(15, 8))
for i, col in enumerate(features_to_scale):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True, color='steelblue')
    plt.title(f"Histogram of {col}")
    plt.xlabel("Standardized Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[features_to_scale + ['mental_health_risk']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()





# Set features and target for classification
X_cls = df.drop(columns=['mental_health_risk'])
y_cls = df['mental_health_risk']
clf = RandomForestClassifier(random_state=42)
clf.fit(X_cls, y_cls)




from sklearn.model_selection import train_test_split

X = df.drop('mental_health_risk', axis=1)
y = df['mental_health_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))



#random forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))


from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
