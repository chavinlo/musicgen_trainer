import torchaudio
from tqdm import trange
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

import os
import typing as tp

import torch

from audiocraft.models.encodec import CompressionModel
from audiocraft.models.lm import LMModel
from audiocraft.models.builders import get_debug_compression_model, get_debug_lm_model
from audiocraft.models.loaders import load_compression_model, load_lm_model
from audiocraft.data.audio_utils import convert_audio
from audiocraft.modules.conditioners import ConditioningAttributes, WavCondition
from audiocraft.utils.autocast import TorchAutocast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=False, default=None)
parser.add_argument('--model_id', type=str, required=False, default='small')
parser.add_argument('--save_path', type=str, required=False, default='test.wav')
parser.add_argument('--duration', type=float, required=False, default=30)
parser.add_argument('--sample_loops', type=int, required=False, default=4)
parser.add_argument('--use_sampling', type=bool, required=False, default=1)
parser.add_argument('--two_step_cfg', type=bool, required=False, default=0)
parser.add_argument('--top_k', type=int, required=False, default=250)
parser.add_argument('--top_p', type=float, required=False, default=0.0)
parser.add_argument('--temperature', type=float, required=False, default=1.0)
parser.add_argument('--cfg_coef', type=float, required=False, default=3.0)
args = parser.parse_args()

model = MusicGen.get_pretrained(args.model_id)

self = model
# print(self.lm.state_dict().keys())
if args.weights_path is not None:
    self.lm.load_state_dict(torch.load(args.weights_path))
attributes, prompt_tokens = self._prepare_tokens_and_attributes([args.prompt], None)
print("attributes:", attributes)
print("prompt_tokens:", prompt_tokens)

duration = args.duration

self.generation_params = {
    'max_gen_len': int(duration * self.frame_rate),
    'use_sampling': args.use_sampling,
    'temp': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'cfg_coef': args.cfg_coef,
    'two_step_cfg': args.two_step_cfg,
}
total = []
for _ in trange(args.sample_loops):
    with self.autocast:
        gen_tokens = self.lm.generate(prompt_tokens, attributes, callback=None, **self.generation_params)
        total.append(gen_tokens[..., prompt_tokens.shape[-1] if prompt_tokens is not None else 0:])
        prompt_tokens = gen_tokens[..., -gen_tokens.shape[-1] // 2:]
gen_tokens = torch.cat(total, -1)

assert gen_tokens.dim() == 3
print("gen_tokens information")
print("Shape:", gen_tokens.shape)
print("Dtype:", gen_tokens.dtype)
print("Contents:", gen_tokens)
with torch.no_grad():
    gen_audio = self.compression_model.decode(gen_tokens, None)
print("gen_audio information")
print("Shape:", gen_audio.shape)
print("Dtype:", gen_audio.dtype)
print("Contents:", gen_audio)
gen_audio = gen_audio.cpu()
torchaudio.save(args.save_path, gen_audio[0], self.sample_rate)
