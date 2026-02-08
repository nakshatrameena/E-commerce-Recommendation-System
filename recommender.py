import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    def __init__(self, products_path, interactions_path):
        self.products = pd.read_csv(products_path)
        self.interactions = pd.read_csv(interactions_path)

        self.products['description'] = self.products['description'].fillna("")

        # ✅ SAFETY: create category column if missing
        if 'category' not in self.products.columns:
            self.products['category'] = "general"
        else:
            self.products['category'] = self.products['category'].fillna("general")

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.products['description']
        )

        self.user_item_matrix = self.interactions.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )

        if len(self.user_item_matrix) > 1:
            self.user_similarity = cosine_similarity(self.user_item_matrix)
        else:
            self.user_similarity = None

    def content_based_recommend(self, product_id, top_n=5):
        matches = self.products[self.products['product_id'] == product_id]
        if matches.empty:
            return []

        idx = matches.index[0]

        cosine_sim = cosine_similarity(
            self.tfidf_matrix[idx],
            self.tfidf_matrix
        ).flatten()

        boosted_scores = []

        for i, score in enumerate(cosine_sim):
            if self.products.iloc[i]['category'] == self.products.iloc[idx]['category']:
                score += 0.2
            boosted_scores.append(score)

        boosted_scores = np.array(boosted_scores)

        similar_indices = boosted_scores.argsort()[::-1]
        similar_indices = similar_indices[similar_indices != idx][:top_n]

        return self.products.iloc[similar_indices][
            ['product_id', 'name']
        ].to_dict(orient='records')

    def user_based_recommend(self, user_id, top_n=5):
        if self.user_similarity is None or user_id not in self.user_item_matrix.index:
            return []

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similar_users = self.user_similarity[user_idx]

        weighted_scores = np.dot(similar_users, self.user_item_matrix.values)
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        weighted_scores[user_ratings > 0] = 0

        top_indices = weighted_scores.argsort()[::-1][:top_n]
        product_ids = self.user_item_matrix.columns[top_indices]

        return self.products[
            self.products['product_id'].isin(product_ids)
        ][['product_id', 'name']].to_dict(orient='records')

    def recommend(self, product_id, user_id=None, top_n=5):
        content_recs = self.content_based_recommend(product_id, top_n)

        if user_id is None:
            return content_recs

        user_recs = self.user_based_recommend(user_id, top_n)

        combined = {rec['product_id']: rec for rec in content_recs}
        for rec in user_recs:
            combined.setdefault(rec['product_id'], rec)

        return list(combined.values())[:top_n]
