import pandas as pd
import requests
import urllib
import json
from urllib.parse import urlencode
import json
import pandas as pd
import os

def dataset_txt_file_download(link, file_name, columns):
  base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
  public_key = link
  final_url = base_url + urlencode(dict(public_key=public_key))
  response = requests.get(final_url)
  download_url = response.json()['href']
  download_response = requests.get(download_url)

  directory = "data"
  if not os.path.exists(directory):
    os.makedirs(directory)

  with open(f"{directory}/{file_name}", 'wb') as f:
    f.write(download_response.content)

  df = pd.read_csv(f"{directory}/{file_name}", sep="\t", header = None)
  df.columns = columns
  pd.DataFrame.to_csv(df, f"data/{file_name[:-4]}.csv")

dataset_txt_file_download('https://disk.yandex.ru/d/KVm7gIom6YooWA', 'Gowalla_totalCheckins.txt', ['user', 'check-in time', 'latitude', 'longitude', 'location_id'])
dataset_txt_file_download('https://disk.yandex.ru/d/Sr3FpAa7WG4GdA', 'Gowalla_edges.txt', ['1st friend', '2nd friend'])

