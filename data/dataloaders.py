from torch.utils.data import Dataset
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice
import requests
import json
import os

class AudioDataset(Dataset):
    def __init__(self, 
                 data_dir
                 ):
        self.data_dir = data_dir
        self.data_map = []

        dir_map = os.listdir(data_dir)
        for d in dir_map:
            name, ext = os.path.splitext(d)
            if ext == '.wav':
                if os.path.exists(os.path.join(data_dir, name + '.txt')):
                    self.data_map.append({
                        "audio": os.path.join(data_dir, d),
                        "label": os.path.join(data_dir, name + '.txt')
                    })
                else:
                    raise ValueError(f'No label file for {name}')
                
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, idx):
        data = self.data_map[idx]
        audio = data['audio']
        label = data['label']

        return audio, label

def AudioWBDS(
            json_map: str,
            base_url: str,
        ):
        
        init_json = requests.get(json_map).json()
        url_list = []

        for key, value in init_json.items():
            #"0.tar": 512,
            url = base_url + key
            url = f"pipe:curl -L -s {url} || true"
            url_list.append(url)

        print("URLs:", len(url_list))
        url_list = url_list
        web_dataset = wds.WebDataset(url_list).decode(
            wds.torch_audio
        )
        
        return web_dataset