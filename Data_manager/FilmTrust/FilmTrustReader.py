#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Maurizio Ferrari Dacrema
"""



import zipfile, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL




class FilmTrustReader(DataReader):
    """
    Not the reference of who proposed the dataset, they simply use it

    @inproceedings{DBLP:conf/ijcai/GuoZY13,
      author    = {Guibing Guo and
                   Jie Zhang and
                   Neil Yorke{-}Smith},
      editor    = {Francesca Rossi},
      title     = {A Novel Bayesian Similarity Measure for Recommender Systems},
      booktitle = {{IJCAI} 2013, Proceedings of the 23rd International Joint Conference
                   on Artificial Intelligence, Beijing, China, August 3-9, 2013},
      pages     = {2619--2625},
      publisher = {{IJCAI/AAAI}},
      year      = {2013},
      url       = {http://www.aaai.org/ocs/index.php/IJCAI/IJCAI13/paper/view/6615},
      timestamp = {Tue, 08 Mar 2022 17:41:51 +0100},
      biburl    = {https://dblp.org/rec/conf/ijcai/GuoZY13.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """
    DATASET_URL = "https://www.librec.net/datasets/filmtrust.zip"
    DATASET_SUBFOLDER = "FilmTrust/"
    AVAILABLE_URM = ["URM_all"]

    IS_IMPLICIT = False

    def __init__(self):
        super(FilmTrustReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path =  self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "filmtrust.zip")

        except (FileNotFoundError, zipfile.BadZipFile):
            self._print("Unable to find or extract data zip file. Downloading...")
            try:
                download_from_URL(self.DATASET_URL, zipFile_path, "filmtrust.zip")
            except:
                zipFile_path = self.DATASET_OFFLINE_ROOT_FOLDER + "FilmTrust/"
            dataFile = zipfile.ZipFile(zipFile_path + "filmtrust.zip")



        URM_path = dataFile.extract("ratings.txt", path=zipFile_path + "decompressed/")

        self._print("Loading Interactions")
        URM_all_dataframe = pd.read_csv(filepath_or_buffer=URM_path, sep=" ", header=None, dtype={0:str, 1:str, 3:float})
        URM_all_dataframe.columns = ["UserID", "ItemID", "Data"]

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset

