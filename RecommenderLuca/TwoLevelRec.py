from Recommenders.BaseRecommender import BaseRecommender

import scipy.sparse as sps
import numpy as np

class TwoLevelRec(BaseRecommender):
    """ TwoLevelRec
    Given two recommenders(recommender1, recommender2), it produces recommendations with recommender1 and 
    ranks them acoordingly to recommender2
    """

    #RECOMMENDER_NAME = "TwoLevelRec"

    def __init__(self, URM_train, recommender1, recommender2, max_cutoff = 350):
        super(TwoLevelRec, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender1 = recommender1
        self.recommender2 = recommender2
        self.RECOMMENDER_NAME = recommender1.RECOMMENDER_NAME + '_' + recommender2.RECOMMENDER_NAME + '_' + "TwoLevelRec"
        self.recommendations = self.recommender1.recommend(list(range(0,self.URM_train.shape[0])), max_cutoff)

        
        
    def fit(self, n_items_to_rank = 100):
        self.n_items_to_rank = n_items_to_rank      
        self.items_to_rank = [items[:n_items_to_rank] for items in self.recommendations]
        #print(self.items_to_rank.shape)


    def set_rec2(self,rec2):
        self.recommender2 = rec2
        self.RECOMMENDER_NAME = self.recommender1.RECOMMENDER_NAME + '_' + self.recommender2.RECOMMENDER_NAME + '_' + "TwoLevelRec"

    def _filterItems(self, user_id, items_scores):

        assert self.URM_train.getformat() == "csr", "TwoLevelRec_Class: URM_train is not CSR, this will cause errors in filtering items"
        all_items_array = np.arange(self.URM_train.shape[1]) 
        items_not_to_rank = np.setdiff1d(all_items_array, self.items_to_rank[user_id])
        items_scores[items_not_to_rank] = -np.inf
        return items_scores


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        
        
        #this could be a loop over a list of pretrained recommender objects
        scores_batch = self.recommender2._compute_item_score(user_id_array, items_to_compute)
       # print(scores_batch.shape)
        #scores_batch2 = self.recommender2._compute_item_score(user_id_array, self.items_to_rank) #non va bene perche items to compute è sottoinsieme di items, a me serve un sottoinsieme diverso per ogni utente non uno per tutti!!
        #print(scores_batch2.shape)
        #print(np.array_equal(scores_batch,scores_batch2))
        for user_index in range(len(user_id_array)):
            user_id = user_id_array[user_index]
            scores_batch[user_index,:] = self._filterItems(user_id, scores_batch[user_index,:])

        return scores_batch




class TwoLevelRec2OPT(BaseRecommender):

    def __init__(self, URM_train, recommender1, recommender2, max_cutoff=350):
        super(TwoLevelRec2OPTlf).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender1 = recommender1
        self.recommender2 = recommender2
        self.RECOMMENDER_NAME = recommender1.RECOMMENDER_NAME + '_' + recommender2.RECOMMENDER_NAME + "_TwoLevelRec2OP      self.recommendations = self.recommender1.recommend(list(range(0, self.URM_train.shape[0])), max_cutoff)"
        
        # Initialize a cache for user scores
        self.scores_cache = {}

    def fit(self, n_items_to_rank=100):
        self.n_items_to_rank = n_items_to_rank      
        self.items_to_rank = [items[:n_items_to_rank] for items in self.recommendations]

    def set_rec2(self, rec2):
        self.recommender2 = rec2
        self.RECOMMENDER_NAME = self.recommender1.RECOMMENDER_NAME + '_' + self.recommender2.RECOMMENDER_NAME + "_TwoLevelRec2OPT   def _filterItems(self, user_id, items_scores):"
        assert self.URM_train.getformat() == "csr", "TwoLevelRec2OPTss: URM_train is not CSR, this will cause errors in filtering items"
        all_items_array = np.arange(self.URM_train.shape[1]) 
        items_not_to_rank = np.setdiff1d(all_items_array, self.items_to_rank[user_id])
        items_scores[items_not_to_rank] = -np.inf
        return items_scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Check if scores for the given users are available in cache
        missing_user_ids = [user_id for user_id in user_id_array if user_id not in self.scores_cache]
        
        # If scores for some users are missing, compute them
        if missing_user_ids:
            new_scores = self.recommender2._compute_item_score(missing_user_ids, items_to_compute)
            for i, user_id in enumerate(missing_user_ids):
                self.scores_cache[user_id] = new_scores[i]

        # Fetch scores either from cache or newly computed scores
        scores_batch = np.vstack([self.scores_cache[user_id] for user_id in user_id_array])
        
        for user_index in range(len(user_id_array)):
            user_id = user_id_array[user_index]
            scores_batch[user_index, :] = self._filterItems(user_id, scores_batch[user_index, :])

        return scores_batch



#In realtà il numero di inference timestep va dato al fit del DifssusionRec
#però per non ritrainarlo da capo ogni volta sarebbe bello poterlo modificare dopo 
"""
class DiffusionTraditionalTwoLevelRec2OPTLevelRec):
        RECOMMENDER_NAME = "TwoLevelRec"

def __init__(self, URM_train, diffusionRec, tradRec):
    super(DiffusionTraditionalTwoLevelRec, self).__init__(URM_train, diffusionRec, tradRec)

    

    
    
def fit(self, n_items_to_rank = 100, inference_timesteps = 100):
    self.n_items_to_rank = n_items_to_rank   
    self.diffusionRec.inference_timesteps = inference_timesteps   
    self.items_to_rank = self.recommender1.recommend(np.arrange(self.URM_train.shape[0]), n_items_to_rank)



class TraditionalDiffusionTwoLevelRec(TwoLevelRec):
        RECOMMENDER_NAME = "TwoLevelRec"

def __init__(self, URM_train, tradRec, tradRec):
    super(TraditionalDiffusionTwoLevelRec, self).__init__(URM_train, diffusionRec, tradRec)

    

    
    
def fit(self, cutoff = 100, inference_timesteps = 100):
    self.cutoff = cutoff   
    self.diffusionRec.inference_timesteps = inference_timesteps   
    self.items_to_rank = self.recommender1.recommend(np.arrange(self.URM_train.shape[0]), cutoff)
"""
