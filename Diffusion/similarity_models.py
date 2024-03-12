import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components

class TwoRandomWalksSimilarity:
    def __init__(self, URM) -> None:
        self.URM = URM
        self.similarity_matrix = self.generate_user_similarity_matrix(URM)

    def get_similarity_matrix(self):
        return self.similarity_matrix

    def urm_to_graph(self, URM):
        """
        Convert a User-Item CSR matrix to a bipartite graph adjacency matrix.
        """
        num_users, num_items = URM.shape
        adjacency_matrix = csr_matrix((num_users + num_items, num_users + num_items))
        adjacency_matrix[:num_users, num_users:] = URM
        adjacency_matrix[num_users:, :num_users] = URM.T
        return adjacency_matrix

    def random_walks(self, adjacency_matrix, num_users):
        """
        Perform random walks on the bipartite graph and return a user-user similarity matrix.
        """
        transition_matrix = diags(1 / np.maximum(adjacency_matrix.sum(axis=1).A.ravel(), 1)).dot(adjacency_matrix)
        user_to_user_matrix = transition_matrix[:num_users, num_users:] @ transition_matrix[num_users:, :num_users]
        return user_to_user_matrix

    def generate_user_similarity_matrix(self, URM):
        graph = self.urm_to_graph(URM)
        num_users = URM.shape[0]
        user_similarity = self.random_walks(graph, num_users)
        return user_similarity

class TwoRandomWalksSampler:
    def __init__(self, URM, warm_user_ids) -> None:
        similarity_model = TwoRandomWalksSimilarity(URM)
        self.n_user = URM.shape[0]
        self.similarity_matrix = similarity_model.get_similarity_matrix()
        self.warm_user_ids = warm_user_ids


    def sample_batch(self, n, user = None):
        random_user = np.random.randint(0, self.n_user) if user == None else user
        similarity_scores = self.similarity_matrix[random_user, :].toarray().flatten()
        
        if n > 1:  
            top_indices = np.argpartition(-similarity_scores, range(n))[:n]  # Select top n similar users, should contain also random user (most similar to itself)
            batch_users = top_indices # np.append(top_indices, random_user)
        else:  
            batch_users = np.array([random_user])
        
        return batch_users
    
    def sample_warm_batch(self, n):
        # Select a random user from the warm user ids
        random_user_idx = np.random.choice(range(self.n_user), 1)[0]
        random_user = self.warm_user_ids[random_user_idx]
        
        # Retrieve similarity scores for the randomly chosen user among warm users
        similarity_scores = self.similarity_matrix[random_user_idx, :].toarray().flatten()
        
        if n > 1:
            # Find top n-1 similar users from the warm users, excluding the randomly chosen user
            top_indices = np.argpartition(-similarity_scores, range(n-1))[:n-1]
            batch_user_idxs = np.append(top_indices, random_user_idx)
        else:
            batch_user_idxs = np.array([random_user_idx])
        
        # Map back the indices to actual user ids
        batch_users = np.array(self.warm_user_ids)[batch_user_idxs]
        
        return batch_users
