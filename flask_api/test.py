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


class Dataset():

  def __init__(self, link, filename, graph_link, graph_filename):
    self.data = self.dataset_txt_file_download(link, filename)
    self.friends = self.dataset_txt_file_download(graph_link, graph_filename)

  def dataset_txt_file_download(self, link, file_name):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = link
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    download_response = requests.get(download_url)

    with open(file_name, 'wb') as f:
      f.write(download_response.content)

    df = pd.read_csv(file_name, sep="\t", header = None)
    return df

  def preprocess(self, num_quantiles, num_users):
    self.data.columns = ['user', 'check-in time', 'latitude', 'longitude', 'location_id']
    self.data['longitude_bin'] = pd.qcut(self.data.longitude, q=num_quantiles, labels = [i for i in range(0, 15)])
    self.data['latitude_bin'] = pd.qcut(self.data.latitude, q=num_quantiles, labels = [i for i in range(0, 15)])
    self.data['longitude_bin'] = self.data['longitude_bin'].astype('int64')
    self.data['latitude_bin'] = self.data['latitude_bin'].astype('int64')
    self.data['location_id_bin'] = list(zip(self.data.latitude_bin, self.data.longitude_bin))
    self.data['place'] = list(zip(self.data.latitude, self.data.longitude))
    self.data['location_id_bin'] = self.data['location_id_bin'].astype('str')

    self.friends.columns = ['1st friend', '2nd friend']

    le = LabelEncoder()
    self.data.location_id_bin = le.fit_transform(self.data.location_id_bin.values)
    self.sectors = self.data.groupby('location_id_bin').agg({'place':lambda x: list(x)}).reset_index()

    self.data = self.data.sort_values('check-in time')
    self.data['check-in time'] = pd.to_datetime(self.data['check-in time'])
    self.data["check-in time"] = self.data['check-in time'].map(lambda x: int(x.timestamp()))

    self.users = self.data.groupby('user').agg({'location_id_bin':lambda x: list(x), 'check-in time':lambda x: list(x)}).reset_index()
    self.users = self.users[self.users.location_id_bin.map(lambda x: len(x) > 3)]

    self.friends = self.friends[self.friends['1st friend'].isin(self.users.user.unique())].reset_index().drop(['index'], axis=1)
    self.friends = self.friends[self.friends['2nd friend'].isin(self.users.user.unique())].reset_index().drop(['index'], axis=1)

    self.reset_id()
    self.reset_edges()
    self.reset_users()

    self.users = self.users[self.users['user'] <= num_users]

    self.friends = self.friends[(self.friends['1st friend'] <= num_users )& (self.friends['2nd friend'] <= num_users)]


  def reset_id(self):
    self.user_id = {}
    id = 0
    for user in self.users.user:
      self.user_id[user] = id
      id += 1
    self.users = self.users.reset_index()

  def reset_edges(self):
    for edge in range(0, self.friends.shape[0]):
      self.friends.loc[edge, '1st friend'] = self.user_id[self.friends.loc[edge, '1st friend']]
    for edge in range(0, self.friends.shape[0]):
      self.friends.loc[edge, '2nd friend'] = self.user_id[self.friends.loc[edge, '2nd friend']]

  def reset_users(self):
    for user in range(0, self.users.shape[0]):
      self.users.loc[user, 'user'] = self.user_id[self.users.loc[user, 'user']]

def gather_user_history(act_list, time_list, n_users, n_bins, n_context):
    user_history = np.zeros((n_users, n_bins, n_context), dtype=np.int32)
    for u in range(0, n_users):
        one_act_list = act_list[u]
        one_time_list = time_list[u]
        for t in range(0, n_bins):
            t_list = [i for i, x in enumerate(one_time_list) if x == t]
            loop_t = t - 1
            if loop_t >= 0:
                while len(t_list) < n_context:
                    temp_list = [i for i, x in enumerate(one_time_list) if x == loop_t]
                    t_list = temp_list + t_list
                    loop_t -= 1
                    if loop_t - 1 < 0:
                        break
            if len(t_list) == 0:
                t_list = [0] * n_context
            now_index = t_list[-n_context:]
            begin_ind = now_index[0]
            end_ind = now_index[-1]
            current_history = one_act_list[begin_ind: end_ind + 1]
            if len(current_history) < n_context:
                current_history = [0] * (n_context - len(current_history)) + current_history
            user_history[u, t, :] = current_history

    return user_history

def load_dataset_timestamp(users, n_context, seq_len):
    act_list = list()
    time_list = list()
    user_list = list()
    max_timestamp = -1.0
    min_timestamp = float('inf')
    for i in range(users.shape[0]):
      t_item_list = list()
      t_time_list = list()

      user = users.loc[i, 'user']
      entries = users.loc[i, 'location_id_bin']
      for j in range(len(entries)):
        item, time_stamp = users.loc[i, 'location_id_bin'][j], users.loc[i, 'check-in time'][j]
        t_item_list.append(int(item))
        t_time_list.append(int(time_stamp))

        if min_timestamp > int(time_stamp):
            min_timestamp = int(time_stamp)
        if max_timestamp < int(time_stamp):
            max_timestamp = int(time_stamp)

      act_list.append(t_item_list[0: seq_len])
      time_list.append(t_time_list[0: seq_len])
      user_list.append(int(user))


    new_time_list = list()
    num_bins = 0


    num_bins = 12
    min_seq_len = 25


    times_bins = np.linspace(min_timestamp, max_timestamp + 1, num=num_bins, dtype=np.int32)
    for a_time_list in time_list:
        temp_time_list = (np.digitize(np.asarray(a_time_list), times_bins) - 1).tolist()
        new_time_list.append(temp_time_list)
  
    n_users = users.shape[0]
    user_history = gather_user_history(act_list, new_time_list, n_users, num_bins, n_context)

    all_examples = []
    for i in range(0, len(act_list)):

        act_seq = act_list[i]
        time_seq = new_time_list[i]
        print(type(act_seq[0]))


        entry = {'items': act_seq,
            'timestamps': time_seq,
            'seq_len': int(len(act_seq)),
            'user' : user_list[i]
        }

        all_examples.append(entry)

    return all_examples, user_history, num_bins
print("Loading dataset...")
dataset = Dataset('https://disk.yandex.ru/d/KVm7gIom6YooWA', 'Gowalla_totalCheckins.txt', 'https://disk.yandex.ru/d/Sr3FpAa7WG4GdA', 'Gowalla_edges.txt')
print("Preprocessing data...")
dataset.preprocess(15, 20001)
data_examples, user_history, num_bins = load_dataset_timestamp(dataset.users, 128, 100)

# with open('flask_api/sectors.pkl', 'rb') as f:
#     sectors = pickle.load(f)

# sectors = {}
# for i in range(dataset.sectors.shape[0]):
#   sectors[dataset.sectors.loc[i, 'location_id_bin']] = dataset.sectors.loc[i, 'place']

header = {'Accept': 'application/json'}
# test_data = {'items': [24, 35, 35, 35, 173, 173, 173, 173, 173, 185, 185, 35],
#  'timestamps': [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
#  'seq_len': 12,
#  'user': 1}
# test_data["sectors"] = sectors
#data_examples[1]["sectors"] = sectors
print("Sent the request")
resp = requests.post("http://0.0.0.0:8000/predict", \
                    json = data_examples[1],\
                    headers= header)
print(resp.status_code)
print(resp.json())
