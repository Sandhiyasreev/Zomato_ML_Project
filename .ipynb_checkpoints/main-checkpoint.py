import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# LOAD DATA
# -----------------------
try:
    df_reviews = pd.read_csv('data/Zomato Restaurant reviews.csv')
    df_meta = pd.read_csv('data/Zomato Restaurant names and Metadata.csv')
    print("df_meta columns:", df_meta.columns.tolist())
    print("✅ Files loaded successfully.\n")
except Exception as e:
    print("❌ Error loading files:", e)
    exit()

# Clean column names
for df in [df_reviews, df_meta]:
    df.columns = df.columns.str.strip()

# -----------------------
# UNIVARIATE ANALYSIS (U)
# -----------------------
plt.figure()
sns.histplot(df_meta['Cost'], kde=True, color='teal')
plt.title("Cost Distribution of Restaurants")
plt.xlabel("Cost")
plt.ylabel("Frequency")
plt.savefig("output_1_cost_distribution.png")
plt.close()

plt.figure()
top_cuisines = df_meta['Cuisines'].value_counts().head(10)
sns.barplot(y=top_cuisines.index, x=top_cuisines.values, palette='viridis')
plt.title("Top 10 Cuisines in Zomato")
plt.xlabel("Count")
plt.ylabel("Cuisine")
plt.savefig("output_2_top_cuisines.png")
plt.close()

# -----------------------
# HANDLE MISSING VALUES
# -----------------------
print("\nMissing values:\n", df_meta.isnull().sum())
df_meta['Cost'] = df_meta['Cost'].replace('[\₹,]', '', regex=True).astype(float)
df_meta.dropna(subset=['Cost', 'Cuisines'], inplace=True)
df_meta = df_meta[df_meta['Cost'] < 2000]

# -----------------------
# FEATURE ENGINEERING (B)
# -----------------------
le = LabelEncoder()
df_meta['Cuisines_encoded'] = le.fit_transform(df_meta['Cuisines'])
features = df_meta[['Cost', 'Cuisines_encoded']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# -----------------------
# CLUSTERING (M)
# -----------------------
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia, marker='o', color='darkorange')
plt.title('Elbow Method - Optimal Cluster Count')
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia')
plt.grid()
plt.savefig("output_3_elbow_method.png")
plt.close()

kmeans = KMeans(n_clusters=4, random_state=0)
df_meta['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure()
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=df_meta['Cluster'], palette='Set1')
plt.title("Restaurant Clusters Based on Features")
plt.xlabel("Avg Cost (scaled)")
plt.ylabel("Cuisines (encoded & scaled)")
plt.savefig("output_4_cluster_visualization.png")
plt.close()

# -----------------------
# SENTIMENT ANALYSIS (U)
# -----------------------
df_reviews['Sentiment Score'] = df_reviews['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df_reviews['Sentiment'] = df_reviews['Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

plt.figure()
sns.countplot(x='Sentiment', data=df_reviews, palette='Set2')
plt.title("Customer Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("output_5_sentiment_distribution.png")
plt.close()

# -----------------------
# ADDITIONAL CHARTS 6-10
# -----------------------
df_meta['Cuisines'].value_counts().head(5).plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
plt.title('Top 5 Cuisines - Share')
plt.ylabel('')
plt.savefig('output_6_top_cuisines_pie.png')
plt.close()

sns.countplot(x='Cluster', data=df_meta, palette='coolwarm')
plt.title('Restaurant Count per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.savefig('output_7_cluster_counts.png')
plt.close()

top_5_cuisines = df_meta['Cuisines'].value_counts().head(5).index
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_meta[df_meta['Cuisines'].isin(top_5_cuisines)], x='Cuisines', y='Cost', palette='Set3')
plt.xticks(rotation=45)
plt.title("Cost Distribution Across Top 5 Cuisines")
plt.savefig("output_8_cost_boxplot_cuisine.png")
plt.close()

sns.violinplot(x='Cluster', y='Cost', data=df_meta, palette='Accent')
plt.title('Cost Distribution by Cluster')
plt.savefig('output_9_violin_cost_cluster.png')
plt.close()

sns.kdeplot(df_reviews['Sentiment Score'], fill=True, color='purple')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.savefig('output_10_kde_sentiment.png')
plt.close()

# -----------------------
# MULTIVARIATE CHARTS 11–13
# -----------------------
sns.scatterplot(data=df_meta, x='Cost', y='Cluster', hue='Cuisines', style='Cluster', palette='tab10', alpha=0.7)
plt.title('Cost vs Cuisines by Cluster')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chart11_cost_vs_cuisines_by_cluster.png")
plt.close()

df_meta['Time Category'] = df_meta['Timings'].apply(lambda x: 'Day' if 'AM' in str(x) else 'Evening/Night')
sns.boxplot(data=df_meta, x='Time Category', y='Cost', hue='Cluster')
plt.title('Cost vs Time Category by Cluster')
plt.savefig("chart12_cost_vs_time_category_by_cluster.png")
plt.close()

cuisine_cluster_df = df_meta.groupby(['Cuisines', 'Cluster']).size().unstack(fill_value=0)
sns.heatmap(cuisine_cluster_df, cmap='YlGnBu', annot=True, fmt='d')
plt.title('Heatmap of Cuisines vs Cluster')
plt.ylabel('Cuisines')
plt.xlabel('Cluster')
plt.tight_layout()
plt.savefig("chart13_heatmap_cuisine_vs_cluster.png")
plt.close()

# -----------------------
# ML MODEL + PERFORMANCE (Charts 14–15)
# -----------------------
df_review_clean = df_reviews.dropna(subset=['Review', 'Sentiment'])
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X = tfidf.fit_transform(df_review_clean['Review'])
y = df_review_clean['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Chart 14
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Logistic Regression")
plt.savefig("chart14_confusion_matrix.png")
plt.close()

# Chart 15
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve((y_test == 'Positive').astype(int), y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Sentiment Classification')
plt.legend(loc="lower right")
plt.savefig("chart15_roc_auc.png")
plt.close()

print("✅ All Charts (1–15) saved successfully. Check .png files in your folder.")
