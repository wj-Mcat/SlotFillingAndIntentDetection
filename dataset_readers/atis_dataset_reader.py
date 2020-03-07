#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :atis_dataset_reader.py
@说明    :
@时间    :2020/03/07 17:14:48
@作者    :吴京京
'''
from allennlp.data.dataset_readers import DatasetReader


@DatasetReader.register("atis")
class ATISDatasetReader(DatasetReader):

    def _read(self, file_path: str):
        pass

