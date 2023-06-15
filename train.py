import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from torch.utils.data import Dataset
    
from audiocraft.modules.conditioners import (
    ClassifierFreeGuidanceDropout
)

import os

import wandb

model = MusicGen.get_pretrained('small')
model.lm = model.lm.to(torch.float32) 

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
    
dataset = AudioDataset('/home/ubuntu/dataset')
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

learning_rate = 0.0001
model.lm.train()

scaler = torch.cuda.amp.GradScaler()

#from paper
optimizer = AdamW(model.lm.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = wandb.init(project='audiocraft')

num_epochs = 10000

inference_step = 50
do_inference = False # doesnt works atm

save_step = 400
save_models = True

def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans

def preprocess_audio(audio_path, model: MusicGen, duration: int = 30):
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    end_sample = int(model.sample_rate * duration)
    wav = wav[:, :end_sample]

    assert wav.shape[0] == 1
    assert wav.shape[1] == model.sample_rate * duration

    wav = wav.cuda()
    wav = wav.unsqueeze(1)

    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio

    assert scale is None

    return codes

def fixnan(tensor: torch.Tensor):
    nan_mask = torch.isnan(tensor)
    result = torch.where(nan_mask, torch.zeros_like(tensor), tensor)
    
    return result

def one_hot_encode(tensor, num_classes=2048):
    shape = tensor.shape
    one_hot = torch.zeros((shape[0], shape[1], num_classes))

    for i in range(shape[0]):
        for j in range(shape[1]):
            index = tensor[i, j].item()
            one_hot[i, j, index] = 1

    return one_hot

duration = 30

use_sampling = False
top_k = 250
top_p = 0.0
temperature = 1.0
cfg_coef = 3.0
two_step_cfg = False

frame_rate = 32000

generation_params = {
    'max_gen_len': int(duration * frame_rate),
    'use_sampling': use_sampling,
    'temp': temperature,
    'top_k': top_k,
    'top_p': top_p,
    'cfg_coef': cfg_coef,
    'two_step_cfg': two_step_cfg,
}

current_step = 0

for epoch in range(num_epochs):
    for batch_idx, (audio, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        #where audio and label are just paths
        audio = audio[0]
        label = label[0]

        audio = preprocess_audio(audio, model) #returns tensor
        text = open(label, 'r').read().strip()

        attributes, _ = model._prepare_tokens_and_attributes([text], None)

        conditions = attributes
        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
        conditions = conditions + null_conditions
        tokenized = model.lm.condition_provider.tokenize(conditions)
        cfg_conditions = model.lm.condition_provider(tokenized)
        condition_tensors = cfg_conditions

        codes = torch.cat([audio, audio], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            lm_output = model.lm.compute_predictions(
                codes=codes,
                conditions=[],
                condition_tensors=condition_tensors
            )

            logits = fixnan(lm_output.logits)
            codes = codes[0]
            probs_codes = one_hot_encode(codes, num_classes=2048)
            logits = logits[0]
            probs_codes = probs_codes.cuda()
            loss = criterion(logits, probs_codes)

        print("Right after compute predictions")
        print(f"Number of nans in logits: {count_nans(lm_output.logits)}")
        print(f"Total number of logits: {lm_output.logits.numel()}")
        nan_percentage = count_nans(lm_output.logits) / lm_output.logits.numel()
        print(f"Percentage of nans: {nan_percentage}")
        print("If less than 25,000, its ok")
        #the nan thing has something to do with special tokens. I think they can be removed but would have to modify compute_predictions functions. Also, it only affects the last bottom layers of the tensor, so it shouldn't have much influence, we just change it to 0 to avoid problems.
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}")
        run.log({
            "loss": loss.item(),
            "nan_percentage": nan_percentage,
            "step": current_step,
        })

        current_step += 1

        if save_models:
            if current_step % save_step == 0:
                torch.save(model.lm.state_dict(), f"saved_models/lm_{current_step}.pt")
