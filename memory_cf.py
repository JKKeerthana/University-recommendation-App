#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MemoryBasedCF:
    """
    Memory-based Collaborative Filtering using cosine similarity.
    The model is built on a pivot table of users and items (universities or specializations) with normalized ratings.
    """
    def __init__(self, pivot):
        self.pivot = pivot
        self.similarity = cosine_similarity(pivot)
        self.user_ids = list(pivot.index)
        self.item_ids = list(pivot.columns)

    def predict(self, input_user):
        if input_user in self.user_ids:
            idx = self.user_ids.index(input_user)
            sim_scores = self.similarity[idx]
        else:
            sim_scores = np.ones(len(self.user_ids)) / len(self.user_ids)
        
        weighted_sum = np.dot(sim_scores, self.pivot.values)
        if weighted_sum.sum() > 0:
            preds = weighted_sum / weighted_sum.sum()
        else:
            preds = weighted_sum
        return preds



# In[ ]:




