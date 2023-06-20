# not suited for average users
# meant for easier understanding of the training process

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
    
from audiocraft.modules.conditioners import (
    ClassifierFreeGuidanceDropout
)

import wandb

from data.dataloaders import AudioWBDS

model = MusicGen.get_pretrained('small')
model.lm = model.lm.to(torch.float32) #important

dataset = AudioWBDS(
    "https://huggingface.co/datasets/atom-in-the-universe/audstock-10k-music/raw/main/train/sizes.json", 
    "https://huggingface.co/datasets/atom-in-the-universe/audstock-10k-music/resolve/main/train/"
    )

eval_dataset = AudioWBDS(
    "https://huggingface.co/datasets/atom-in-the-universe/audstock-10k-music/raw/main/test/sizes.json",
    "https://huggingface.co/datasets/atom-in-the-universe/audstock-10k-music/resolve/main/test/"
)

train_dataloader = DataLoader(dataset, batch_size=1)
eval_dataloader = DataLoader(eval_dataset, batch_size=1)

learning_rate = 0.0001
model.lm.train()

scaler = torch.cuda.amp.GradScaler()

#from paper
optimizer = AdamW(model.lm.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = wandb.init(project='audiocraft')

num_epochs = 10000

save_step = 200
eval_step = 25
save_models = True

def count_nans(tensor):
    nan_mask = torch.isnan(tensor)
    num_nans = torch.sum(nan_mask).item()
    return num_nans

def preprocess_audio(audio_tensor, model: MusicGen, duration: int = 30):
    wav, sr = audio_tensor

    #tmp
    wav: torch.Tensor
    wav = wav.squeeze(0)

    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    end_sample = int(model.sample_rate * duration)
    wav = wav[:, :end_sample]

    # pad if missing
    if wav.shape[1] < model.sample_rate * duration:
        wav = torch.nn.functional.pad(wav, (0, model.sample_rate * duration - wav.shape[1]))

    #print("Shape", wav.shape)

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

current_step = 0

separator = ", "

for epoch in range(num_epochs):
    for batch_idx, contents in enumerate(train_dataloader):
        optimizer.zero_grad()

        #where audio and label are just paths
        audio = contents['flac'] # tensor with wav and sr
        text = contents['json']['text'][0][0] # string

        for tag in contents['json']['tag']:
            text += separator + tag[0]

        audio = preprocess_audio(audio, model) #returns tensor

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

        print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item()}")
        run.log({
            "loss": loss.item(),
            "step": current_step,
            "epoch": epoch
        })

        current_step += 1

        if current_step % eval_step == 0:

            loss = torch.tensor(0.0).cuda()

            total_evals = 0

            with torch.no_grad():
                for batch_idx, contents in enumerate(eval_dataloader):
                    #where audio and label are just paths
                    audio = contents['flac'] # tensor with wav and sr
                    text = contents['json']['text'][0][0] # string

                    for tag in contents['json']['tag']:
                        text += separator + tag[0]

                    audio = preprocess_audio(audio, model)

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

                        loss = loss + criterion(masked_logits,masked_codes)

                        total_evals = total_evals + 1

                        print(f"Eval Batch: {batch_idx}, Loss: {loss.item() / total_evals}")

                        if total_evals >= 10:
                            break
                
                loss = loss / total_evals
                print(f"Eval Loss: {loss.item()}")
                run.log({
                    "eval_loss": loss.item(),
                    "epoch": epoch
                })

        if save_models:
            if current_step % save_step == 0:
                torch.save(model.lm.state_dict(), f"saved_models/lm_{current_step}.pt")