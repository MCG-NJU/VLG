import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model

from datasets import build_dataset
from losses import DistillationLoss, PretrainSentLoss, LabelSmoothingCrossEntropy, KLLoss
from samplers import RASampler, WeightedDistributedSampler
from engine import calc_class_acc_dist, evaluate_LT_dist, evaluate_pretrain_dist, train_one_epoch, select_sent_dist
from engine_openmax import evaluate_openset, evaluate_openset_v2, evaluate_openset_dist, evaluate_openset_v2_dist
from optim_factory import create_optimizer
from mixup import Mixup
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributed as dist
from solver import _optimizer, _lr_scheduler

import utils
import os.path as osp
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('VLG training and evaluation script for open set', add_help=False)
    parser.add_argument('--fp32-resume', action='store_true', default=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--config', required=True, type=str, help='config')
    parser.add_argument('--pretrained-bert', default=None, type=str)
    parser.add_argument('--txt-embed-path', type=str, default=None, help='config')
    parser.add_argument('--vis-backbone-path', type=str, default=None, help='config')
    parser.add_argument('--two-branch', action='store_true', help='two branch output')
    parser.set_defaults(two_branch=False)
    parser.add_argument('--debug', action='store_true', help='cls and img txt contrastive learning')
    parser.set_defaults(debug=False)

    # NLP parameters
    parser.add_argument('--desc-path', default='', type=str)
    parser.add_argument('--context-length', default=0, type=int, help='max length of text description')
    parser.add_argument('--sent-length', default=64, type=int, help='max number of selected sentences')
    parser.add_argument('--cls-token-length', default=1, type=int, help='the length of cls token')
    parser.add_argument('--loss-type', default='CE', type=str, help='loss type')
    parser.add_argument('--pretrain-cvlp', action='store_true', help='sentence-level pretraining')
    parser.set_defaults(pretrain_cvlp=False)
    parser.add_argument('--pretrain-cvlp-path', default='', type=str,
                        help='path of sentence-level pretraining task ckpt')

    # Model parameters
    parser.add_argument('--model', default='pvt_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--scale-size', default=256, type=int, help='images scale size')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--img-grad', action='store_true', default=True)
    parser.add_argument('--no-img-grad', action='store_false', dest='img_grad')
    parser.add_argument('--train-mode', action='store_true', default=True)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--text-lr', type=float, default=0, metavar='LR',
                        help='learning rate for text model (default: 0)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--clip-ms', action='store_true', help='use clip mean & std for initialization')
    parser.set_defaults(clip_ms=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default=None, type=str, metavar='MODEL',
                        help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str, default=None)
    parser.add_argument('--distillation-type', default='none', choices=['none', 'feat', 'logits', 'logits_kl'],
                        type=str, help="")
    parser.add_argument('--distillation-alpha', default=0, type=float, help="")
    parser.add_argument('--distillation-beta', default=0, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('--distillation-training-mode', action='store_true', help="")
    parser.set_defaults(distillation_training_mode=False)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--pretrained', action='store_true')

    # Sampler Parameters
    parser.add_argument('--weight-sample', action='store_true')
    parser.add_argument('--no-weight-sample', action='store_false', dest='weight_sample')
    parser.set_defaults(weight_sample=False)
    parser.add_argument('--use-sqrt-freq', action='store_true')
    parser.set_defaults(use_sqrt_freq=False)

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='Kinetics', choices=['UCF101', 'HMDB51', 'Kinetics',
                                                                "Kinetics100_base", 'Kinetics100_test', 'k100_support_query',
                                                                'kinetics400_openset'],
                        type=str, help='Kinetics dataset path')

    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only', default=False)
    parser.add_argument('--test', action='store_true', help='Perform test only', default=False)
    parser.set_defaults(test=False)
    parser.add_argument('--test-p', action='store_true', help='Calculate acc for each class', default=False)
    parser.add_argument('--select', action='store_true', help='Perform test only', default=False)
    parser.set_defaults(select=False)
    parser.add_argument('--eval-pretrain', action='store_true', help='Perform evaluation for pretraining')
    parser.set_defaults(eval_pretrain=False)
    parser.add_argument('--ensemble', action='store_true',
                        help='Perform zero-shot evaluation for pretraining like CLIP')
    parser.set_defaults(ensemble=False)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--drop-last', action='store_true')
    parser.add_argument('--no-drop-last', action='store_false', dest='drop_last')
    parser.set_defaults(drop_last=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.set_defaults(broadcast_bn_buffer=True)
    parser.set_defaults(use_actionclip_solver=False)
    parser.add_argument('--randaug-n', type=int, default=0)
    parser.add_argument('--randaug-m', type=int, default=0)
    parser.add_argument('--f-ratio', type=float, default=10)
    parser.add_argument('--ratio', type=float, default=1)
    parser.set_defaults(only_fusion_module=False)
    parser.add_argument('--only-fusion-module', type=bool, default=False)
    parser.set_defaults(freeze_visual=False)
    parser.add_argument('--freeze-visual', type=bool, default=False)
    parser.set_defaults(pretrained_weight=False)
    parser.set_defaults(dense_sample=False)
    parser.add_argument('--dense-sample', action='store_true')
    parser.add_argument('--test-crops', type=int, default=1)
    parser.add_argument('--num-clips', type=int, default=10)
    parser.add_argument('--test-batch-size', type=int, default=0)
    parser.add_argument('--use-softmax', action='store_true', default=False)
    parser.add_argument('--twice-sample', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--use-res', type=bool, default=False)
    parser.add_argument('--no-fusion', type=bool, default=True)

    parser.add_argument('--val-interval', type=int, default=-1)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--fusion-layers', type=int, default=6)
    parser.add_argument('--without-dist-loss', type=bool, default=False)
    parser.add_argument('--naive-txt', action='store_true', default=False)

    parser.add_argument('--weibull-tail', default=20, type=int, help='Classes used in testing')
    parser.add_argument('--weibull-alpha', default=3, type=int, help='Classes used in testing')
    parser.add_argument('--weibull-threshold', default=0.9, type=float, help='Classes used in testing')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(split="train", args=args)
    if args.test:
        dataset_test, _ = build_dataset(split="test", args=args)
    dataset_val, _ = build_dataset(split="val", args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.dist_eval:
        sampler_train = torch.utils.data.DistributedSampler(
                dataset_train,
                num_replicas=num_tasks,
                rank=global_rank, shuffle=True,
                drop_last=args.drop_last)
        if args.test:
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test,
                rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=args.drop_last,
    )

    if args.test:
        if args.test_batch_size == 0:
            test_batch_size = int(args.batch_size * 1.5)
        else:
            test_batch_size = int(args.test_batch_size)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        dataset=dataset_train,
        args=args)

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.resume:
        if args.resume.endswith('RN50.pt'):
            checkpoint = {}
            checkpoint['model'] = torch.jit.load(args.resume, map_location='cpu').state_dict()
        elif args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif osp.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
        else:
            checkpoint = None

        if checkpoint is not None:
            if 'model' in checkpoint:
                msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            else:
                msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
            print(msg)

    if args.eval:
        openset_type = args.openset_type
        if openset_type == 1:
            if args.dist_eval:
                evaluate_openset_dist(data_loader_train, data_loader_test, model, device, args=args,
                                 tokens=None)
            else:
                evaluate_openset(data_loader_train, data_loader_test, model, device, args=args,
                                 tokens=None)
        elif openset_type == 2:
            if args.dist_eval:
                evaluate_openset_v2_dist(data_loader_train, data_loader_test, model, device, args=args,
                                    tokens=None)
            else:
                evaluate_openset_v2(data_loader_train, data_loader_test, model, device, args=args,
                                    tokens=None)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VL-LTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
