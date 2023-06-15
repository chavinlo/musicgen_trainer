import torchaudio
from audiocraft.models import MusicGen
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

def train(
        dataset_path: str,
        model_id: str,
        lr: float,
        epochs: int,
        use_wandb: bool,
        save_step: int = None,
):
    
    if use_wandb is True:
        import wandb
        run = wandb.init(project='audiocraft')

    model = MusicGen.get_pretrained(model_id)
    model.lm = model.lm.to(torch.float32) #important
        
    dataset = AudioDataset(dataset_path)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    learning_rate = lr
    model.lm.train()

    scaler = torch.cuda.amp.GradScaler()

    #from paper
    optimizer = AdamW(model.lm.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = epochs

    save_step = save_step
    save_models = False if save_step is None else True

    save_path = "models/"

    os.makedirs(save_path, exist_ok=True)

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

                codes = codes[0]
                logits = lm_output.logits[0]
                mask = lm_output.mask[0]

                codes = one_hot_encode(codes, num_classes=2048)

                codes = codes.cuda()
                logits = logits.cuda()
                mask = mask.cuda()

                mask = mask.view(-1)
                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

                loss = criterion(masked_logits,masked_codes)

            assert count_nans(masked_logits) == 0
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}")

            if use_wandb is True:
                run.log({
                    "loss": loss.item(),
                    "step": current_step,
                })

            current_step += 1

            if save_models:
                if current_step % save_step == 0:
                    torch.save(model.lm.state_dict(), f"{save_path}/lm_{current_step}.pt")

    torch.save(model.lm.state_dict(), f"{save_path}/lm_final.pt")