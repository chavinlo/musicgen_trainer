from train import train

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--model_id', type=str, required=False, default='small')
parser.add_argument('--lr', type=float, required=False, default=1e-5)
parser.add_argument('--epochs', type=int, required=False, default=100)
parser.add_argument('--use_wandb', type=int, required=False, default=0)
parser.add_argument('--save_step', type=int, required=False, default=None)
parser.add_argument('--no_label', type=int, required=False, default=0)
parser.add_argument('--tune_text', type=int, required=False, default=0)
parser.add_argument('--weight_decay', type=float, required=False, default=1e-5)
parser.add_argument('--grad_acc', type=int, required=False, default=2)
parser.add_argument('--warmup_steps', type=int, required=False, default=16)
parser.add_argument('--batch_size', type=int, required=False, default=4)
parser.add_argument('--use_cfg', type=int, required=False, default=0)
args = parser.parse_args()

train(
    dataset_path=args.dataset_path,
    model_id=args.model_id,
    lr=args.lr,
    epochs=args.epochs,
    use_wandb=args.use_wandb,
    save_step=args.save_step,
    no_label=args.no_label,
    tune_text=args.tune_text,
    weight_decay=args.weight_decay,
    grad_acc=args.grad_acc,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    use_cfg=args.use_cfg,
)
