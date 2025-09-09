import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


data_dir = Path("data")
# Load data
old_df = pd.read_csv(data_dir / "kaggle_arxiv_dataset" / "dataset.csv")


# Combine title and abstract
def combine_text(df):
    return (df["title"] + " " + df["abstract"]).fillna("").tolist()

def remove_stops(text, stops):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stops]
    return " ".join(tokens)


# Preprocessing
def preprocess(texts) -> list:
    # get stopwords
    stop_words = set(stopwords.words("english"))
    # Preprocess
    preprocessed = []
    for text in texts:
        processed_text = remove_stops(text, stop_words)
        preprocessed.append(processed_text)
    return preprocessed

old_combined_texts = combine_text(old_df)
old_texts = preprocess(old_combined_texts)

# print(old_combined_texts[0][:90])
# print(old_texts[0][:90])

vectorizer = TfidfVectorizer(lowercase=True,
                             max_features=100,
                             max_df=0.8,
                             min_df=5,
                             ngram_range=(1,3),
                             stop_words="english",
                             )

vectors = vectorizer.fit_transform(old_texts)

feature_names = vectorizer.get_feature_names_out()

dense = vectors.todense()
denselist = dense.tolist()

all_keywords = []

for description in denselist:
    x = 0
    keywords = []
    for word in description:
        if word > 0:
            keywords.append(feature_names[x])
        x += 1
    all_keywords.append(keywords)

print(old_texts[0])
print(all_keywords[0])


true_k = 5
model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
model.fit(vectors)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

with open(data_dir / "trc_results.txt", 'w', encoding="utf-8") as f:
    for i in range(true_k):
        f.write(f"Cluster {i}\n")
        for ind in order_centroids[i,:10]:
            f.write('    %s\n' % terms[ind],)
        f.write("\n\n")

kmean_indices = model.fit_predict(vectors)

pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

colors = ['r', 'b', 'c', 'y', 'm']

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fix, ax = plt.subplots(figsize=(50,50))

ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

# for i, txt in enumerate(names)
#     ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))

plt.savefig(data_dir / "trc.png")