# -*- coding: utf-8 -*-
import os, sys, inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
import zipfile

import numpy as np
from sklearn.externals import joblib
from sklearn.pipeline import FeatureUnion, Pipeline

from modules.tokenizer import *
from modules.util import *
_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
def get_data(path):
    return os.path.join(_ROOT, 'src', path)
    
class Classifier:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.vectorizer = None
        self.load()

    def load(self):
        zipname = get_data('data.zip')
        ziptarget = get_data('')

        if(len(os.listdir(ziptarget)) == 1):
            with zipfile.ZipFile(zipname, 'r') as z:
                z.extractall(ziptarget)
        filename = get_data('Random Forest Grid Search Noise 0.8 LDA 1000.pkl')
        loaded_model = joblib.load(filename)
        self.model = loaded_model

        vectorizername = get_data('vectorizer.pkl')
        vectorizer_model = joblib.load(vectorizername)
        vectorizer_model.tokenizer = process_text
        self.vectorizer = vectorizer_model

    def classify(self, params):
        self.build_pipeline()

        # must be a numpy array
        if(type(params) != np.ndarray):
            params = np.array(params, dtype=np.object)

        # TRUE, FALSE
        return self.pipeline.predict_proba(params)


    def build_pipeline(self, n_components=2000):
        # vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords, min_df=1, lowercase=True, strip_accents='unicode')

        pipeline_union = Pipeline(
            [
                ('features', FeatureUnion(
                    [
                        ('text', Pipeline(
                            [
                                ('extract', ColumnExtractor([...,0])),
                                ('vectorize', self.vectorizer),
                                ('reduce_dim', TruncateLDA(num_topics = n_components))
                            ]
                        )),
                        ('no_text', Pipeline(
                            [
                                ('extract', ColumnExtractor([...,slice(1,None,None)], datatype=np.float64))
                            ]
                        ))
                    ]
                )),
                self.model.steps[0]
            ]
        )


            # Pipeline([
            # ('vectorize', self.vectorizer),
            # ('reduce_dim', TruncateLDA(num_topics = n_components)),
            # self.model.steps[0]
            # ])

        self.pipeline = pipeline_union

def main():
    cl = Classifier();
    cl.classify([
                 ['@indiferencia @cnagency: cobertura en vivo del terremoto de chile por ustream: http://bit.ly/bnjhzj // lo estoy viendo ahora', 126, 114],
                 ['rt @montesdetoledo: el terremoto que ha sacudido chile es el 2º mayor en los últimos 20 años en el mundo: http://bit.ly/96ydqb #terremot', 477, 265],
                 ['arjona', 0, 0],
                 ['chamoooo el temblor se produjo fue en el oceano pacificooo 90 km cerca de la ciudad de santiagoo tan (cont) http://tl.gd/c6nbi', 496, 161],
                 ['Buscamos a Thomas Born perdido en el terremoto del pasado 27', 1000, 1000],
                 ['buenas nocjhes me voy a casa',0,27]
    ])

if __name__ == "__main__": main()