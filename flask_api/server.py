import pandas as pd
import requests
import urllib
import json
from urllib.parse import urlencode
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
from dateutil.parser import parse
import pickle
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
import pandas as pd
from flask import Flask, jsonify, request


class RecModel(nn.Module):
  def __init__(self):
    super(RecModel, self).__init__()
    self.fc1 = nn.Linear(128, 186)
    self.rnn = nn.RNN(128, 128, batch_first = True)
    self.norm = nn.BatchNorm1d(128)

  def forward(self, x, seq_len):

    x, h = self.rnn(x)
    hx = torch.zeros(x.shape[0], x.shape[2])
    for i in range(hx.shape[0]):
      hx[i] = x[i][seq_len - 1]
    # hx = self.norm(hx)

    x = self.fc1(hx)

    return x
  
with open('sectors.pkl', 'rb') as f:
    sectors = pickle.load(f)


class Recommender():

  def __init__(self, num_classes):
    self.item_emb  = nn.init.xavier_uniform_(torch.empty(num_classes, 128))


  def get_rec(self, user_json):
    print(user_json)
    comb_input = np.concatenate([np.expand_dims(user_json['items'], axis=-1),
                                                np.expand_dims(user_json['timestamps'], axis=-1)], axis=1)

    input_emb = self.item_emb[comb_input[:, 0]]
    input_emb = input_emb.unsqueeze(0)
    rec_model = RecModel()

    # load saved model
    rec_model.load_state_dict(torch.load('models/rec_model.pth'))
    probs = rec_model(input_emb, user_json['seq_len'])
    prediction = torch.argmax(probs, axis = 1).item()
    places = self.get_places(prediction, user_json)

    return self.rec_to_json(places, user_json['user'])

  def rec_to_json(self, places, user_id):
    rec_answer = {}
    rec_answer['user'] = user_id
    rec_answer['recommendations'] = []
    for place in places:
      rec_answer['recommendations'].append({'latitude' : place[0], 'longitude' : place[1]})
    return rec_answer


  def get_places(self, prediction, user_json):
    sector = sectors[prediction]
    indices = np.arange(len(sector))
    random_indices = random.choices(indices, k=10)

    places = []
    for index in random_indices:
      places.append(sector[index])
    return places
  



app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def apicall():
    try:
        test_json = request.get_json()
        print('Test json', test_json)
        test = test_json#pd.read_json(test_json, orient='records')

    except Exception as e:
        raise e

    else:
        recommender = Recommender(186)
        recommendation = recommender.get_rec(test)
        print("Recommendations are ready")
        responses = recommendation

        return (responses)