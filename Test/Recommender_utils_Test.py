#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/09/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import similarityMatrixTopK

import numpy as np
import scipy.sparse as sps
import unittest


class MyTestCase(unittest.TestCase):

    def test_similarityMatrixTopK_denseToDense(self):

        numRows = 100

        TopK = 20

        dense_input = np.random.random((numRows, numRows))
        dense_output = similarityMatrixTopK(dense_input, k=TopK)

        numExpectedNonZeroCells = TopK*numRows

        numNonZeroCells = np.sum(dense_output!=0)

        self.assertEqual(numExpectedNonZeroCells, numNonZeroCells, "DenseToDense incorrect")


    def test_similarityMatrixTopK_sparseToSparse(self):

        numRows = 20

        TopK = 5

        dense_input = np.random.random((numRows, numRows))

        topk_on_dense_input = similarityMatrixTopK(dense_input, k=TopK)

        sparse_input = sps.csc_matrix(dense_input)
        topk_on_sparse_input = similarityMatrixTopK(sparse_input, k=TopK)

        topk_on_dense_input = topk_on_dense_input.toarray()
        topk_on_sparse_input = topk_on_sparse_input.toarray()

        self.assertTrue(np.allclose(topk_on_dense_input, topk_on_sparse_input), "sparseToSparse CSC incorrect")





    def test_similarityMatrixTopK(self):

        item_item_similarity = np.array(
           [[1, 3, 5, 7, 9],
            [3, 5, 7, 9, 1],
            [5, 7, 9, 1, 3],
            [7, 9, 1, 3, 5],
            [9, 1, 3, 5, 7],]
        )

        item_item_similarity = sps.csr_matrix(item_item_similarity)

        topK = 3

        topk_on_dense_input = similarityMatrixTopK(item_item_similarity, k=topK).toarray()


        item_item_similarity_expected = np.array(
           [[0, 0, 5, 7, 9],
            [0, 5, 7, 9, 0],
            [5, 7, 9, 0, 0],
            [7, 9, 0, 0, 5],
            [9, 0, 0, 5, 7],]
        )

        self.assertTrue(np.allclose(topk_on_dense_input, item_item_similarity_expected))







if __name__ == '__main__':

    unittest.main()

