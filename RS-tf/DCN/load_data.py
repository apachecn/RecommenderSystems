import pandas as pd
import numpy as np


class FeatureDict(object):
    def __init__(self, trainfile=None, testfile=None,numeric_cols=[],ignore_cols=[],cate_cols=[]):
        self.trainfile = trainfile
        self.testfile = testfile
        self.cate_cols = cate_cols
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()
     
    '''
    Generate categorical deature dict
    ex: df[col1] = [3,4,1,0,2];df[col2] = [-1,2,7]
    generated feat_dict = {'col1':{3:0,4:1,1:2,0:3,2:4},'col2':{-1:5,2:6,7:7}}
    '''
    def gen_feat_dict(self):
        df = pd.concat([self.trainfile,self.testfile])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols or col in self.numeric_cols:
                continue
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc
        
        
        
class DataPaser(object):
    def __init__(self,feat_dict):
        self.feat_dict = feat_dict
    
    def parse(self,df=None,has_label=False):
        dfi = df.copy() # feature index
        if has_label:
            y = dfi['target'].values.tolist()
            dfi.drop(['id','target'], axis=1, inplace=True)
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'],axis=1,inplace=True)
        
        numeric_values = dfi[self.feat_dict.numeric_cols].values.tolist()
        dfi.drop(self.feat_dict.numeric_cols, axis=1,inplace=True)
        
        dfv = dfi.copy() # dfv for feature values which binary or float
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1,inplace=True)
                dfv.drop(col, axis=1,inplace=True)
                continue
            # categories feature
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.
                
        cate_idx = dfi.values.tolist()
        cate_values = dfv.values.tolist()
        if has_label:
            return cate_idx, cate_values, numeric_values, y
        else:
            return cate_idx, cate_values, numeric_values, ids