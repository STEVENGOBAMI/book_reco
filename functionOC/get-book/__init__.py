import logging

import os
import random

import pandas as pd
import numpy as np
from collections import defaultdict

import pickle

import azure.functions as func

import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
sys.path.append(os.path.dirname(__file__))
print(sys.path, file=sys.stdout)

articles_df = pd.read_csv('model_df/articles_metadata.csv')
pkl_filename = "model_df/pickle_surprise_model_KNNWithMeans.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

def predict_top_five(user_id, model, article_df):
    predictions = {}
    
    #Category 1 to 460
    for i in range(1, 460):
        _, cat_id, _, est, err = model.predict(user_id, i)
        
        #Keep prediction only if we could keep it.
        if (err != True):
            predictions[cat_id] = est
    
    best_cats_to_recommend = dict(sorted(predictions.items(),
                                         key=lambda x: x[1], reverse=True)[:5])
    
    recommended_articles = []
    for key, _ in best_cats_to_recommend.items():
        recommended_articles.append(int(article_df[article_df['category_id']
                                                    == key]['article_id'].sample(1).values))
    
    #return random_articles_for_best_cat, best_cat_to_recommend
    return recommended_articles, best_cats_to_recommend

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    # print(req.get_json())
    name = req.route_params.get("userId") 

    results, recommended_cats = predict_top_five(name, model, articles_df)

    return func.HttpResponse(f'{results}')

