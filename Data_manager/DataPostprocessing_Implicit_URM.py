#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/01/2018

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.DataPostprocessing import DataPostprocessing
import numpy as np



class DataPostprocessing_Implicit_URM(DataPostprocessing):
    """
    This class transforms the URM from explicit (or whatever data content it had) to implicit.
    All interaction having a value >= of the greater_equal_than attribute will be set to 1
    the others to zero.
    """

    def __init__(self, dataReader_object, greater_equal_than):

        assert greater_equal_than >= 0,\
            "DataPostprocessing_Implicit_URM: greater_equal_than must be a positive value >= 0, provided value was {}".format(greater_equal_than)

        super(DataPostprocessing_Implicit_URM, self).__init__(dataReader_object)

        self.greater_equal_than = greater_equal_than

    def is_implicit(self):
        return True

    def _get_dataset_name_data_subfolder(self):
        """
        Returns the subfolder inside the dataset folder tree which contains the specific data to be loaded
        This method must be overridden by any data post processing object like k-cores / user sampling / interaction sampling etc
        to be applied before the data split

        :return: original or k_cores etc...
        """

        subfolder_name = "implicit_ge_{}/".format(self.greater_equal_than)

        inner_subfolder_name = self.dataReader_object._get_dataset_name_data_subfolder()

        # Avoid concatenating the original/ part
        if inner_subfolder_name != self.DATASET_SUBFOLDER_ORIGINAL:
            subfolder_name += inner_subfolder_name

        return subfolder_name



    def _replace_interactions_with_ones(self, loaded_dataset):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos= 0

        blockSize = 1000

        for URM_name, URM_obj in loaded_dataset.AVAILABLE_URM.items():

            while end_pos < len(URM_obj.data):

                end_pos = min(len(URM_obj.data), end_pos + blockSize)
                URM_obj.data[start_pos:end_pos] = URM_obj.data[start_pos:end_pos] >= self.greater_equal_than

                start_pos += blockSize

            URM_obj.eliminate_zeros()



    def _load_from_original_file(self):
        """
        _load_from_original_file will call the load of the dataset and then apply on it the k-cores
        :return:
        """

        loaded_dataset = self.dataReader_object.load_data()
        self._replace_interactions_with_ones(loaded_dataset)

        return loaded_dataset


