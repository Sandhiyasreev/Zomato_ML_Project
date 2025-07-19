import matplotlib.pyplot as plt
import seaborn as sns

def plot_rating_distribution(df):
    sns.histplot(df['rating'], kde=True)
    plt.title('Distribution of Ratings')
    plt.show()
    # Add business insight comment here

# Add other chart functions following UBM rule
