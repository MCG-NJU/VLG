import argparse
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.utils import accuracy

from fewshot_datasets import *
import torch.nn.functional as F
from engine import calc_class_acc_dist, evaluate_LT_dist, evaluate_pretrain_dist, train_one_epoch, select_sent_dist

import utils
import os.path as osp
import warnings

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Few shot script', add_help=False)
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
                                                                   "Kinetics100_base", 'Kinetics100_test',
                                                                   'k100_support_query',
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

    parser.add_argument('--val-interval', type=int, default=-1)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--fusion-layers', type=int, default=6)
    parser.add_argument('--without-dist-loss', type=bool, default=False)
    parser.add_argument('--naive-txt', action='store_true', default=False)
    return parser


def fewshot_test(test_loader, test_datamgr, pretrain_model, args):
    device = torch.device(args.device)
    full_text_tokens = test_datamgr.full_text_tokens
    full_end_idx = test_datamgr.full_end_idxs
    n_support = args.n_support
    acc0_all = []
    acc1_all = []

    with torch.no_grad():
        for i, (all_images, all_targets) in enumerate(test_loader):
            all_images = all_images.to(device)
            all_targets = all_targets.to(device)
            n_way, sq = all_targets.shape
            n_query = sq - n_support

            old_to_new = dict()
            old_labels = all_targets[:, 0].cpu().numpy()
            for new, old in enumerate(old_labels):
                old_to_new[old] = new
            text_tokens, end_idxs = [], []
            for old in old_labels:
                text_tokens.append(full_text_tokens[old])
                assert len(full_text_tokens[old]) == full_end_idx[old]
                end_idxs.append(full_end_idx[old])

            all_targets = torch.tensor(range(n_way)).unsqueeze(dim=-1).to(device)
            all_targets = all_targets.expand(n_way, sq)

            support_images = all_images[:, :n_support]
            support_targets = all_targets[:, :n_support]
            query_images = all_images[:, n_support:]
            query_targets = all_targets[:, n_support:]

            # select
            assert text_tokens is not None and isinstance(text_tokens, list), \
                "text_tokens is None, This function only supports pretraining phase"
            text_tokens = torch.cat(text_tokens)
            sent_idxs = end_idxs
            text_targets = torch.empty((sum(sent_idxs),), dtype=torch.long).to(device)
            left = 0
            for i in range(len(sent_idxs)):
                text_targets[left: left + sent_idxs[i]] = i
                left += sent_idxs[i]

            support_images = support_images.reshape((n_way * n_support, args.num_segments, 3) +
                                                    support_images.size()[-2:])
            support_images = support_images.view((-1, 3) + support_images.size()[-2:])
            with torch.cuda.amp.autocast():
                image_features = pretrain_model.encode_image(support_images)
                identity = image_features
                if hasattr(pretrain_model, 'fusion_model'):
                    image_features = image_features.view(n_way * n_support, args.num_segments, -1)
                    image_features = pretrain_model.fusion_model(image_features)
                    if args.use_res:
                        assert pretrain_model.use_res == args.use_res
                        identity = identity.reshape(
                            (identity.shape[0] // args.num_segments, args.num_segments, -1)).mean(dim=1, keepdim=False)
                        image_features = image_features + identity
                else:
                    image_features = image_features.reshape((n_way * n_support, -1) + image_features.shape[1:]). \
                        mean(dim=1, keepdim=False)
                image_embeddings = image_features
                image_targets = support_targets.reshape(-1)

            batch_tokens = text_tokens.to(device)
            with torch.cuda.amp.autocast():
                text_features = pretrain_model.encode_text(batch_tokens)
                text_embeddings = text_features
                saved_text_embeddings = text_embeddings

            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            text_ces = []
            iter = range(text_embeddings.shape[0])
            for i in iter:
                text_embedding = text_embeddings[i]
                logit = text_embedding @ image_embeddings.t() * pretrain_model.logit_scale.exp()
                label = image_targets == text_targets[i]
                label = label / label.sum()
                ce = torch.sum(-label * F.log_softmax(logit, dim=-1), dim=-1)
                text_ces.append(ce)
            text_ces = torch.tensor(text_ces).to(device)
            print('done select')

            # finetune model
            finetune_model = create_model(
                args.finetune_model,
                pretrained=args.pretrained,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                sent_idxs=end_idxs,
                args=args)

            vis_backbone_path = getattr(args, 'vis_backbone_path',
                                        osp.join(args.pretrain_cvlp_path, "checkpoint.pth"))
            if vis_backbone_path is None:
                vis_backbone_path = osp.join(args.pretrain_cvlp_path, "checkpoint.pth")
            finetune_model._use_txt_ce(saved_text_embeddings, text_ces)
            finetune_model.text_embeddings.requires_grad_(False)
            finetune_model._load_vis_backbone(vis_backbone_path)
            finetune_model.visual.requires_grad_(False)
            finetune_model.text_block.attn.requires_grad_(False)
            if hasattr(pretrain_model, 'fusion_model'):
                finetune_model.fusion_model.requires_grad_(False)

            finetune_model.to(device)

            # finetune result
            query_images.to(device, non_blocking=True)
            query_targets.to(device, non_blocking=True)
            query_images = query_images.reshape((n_way * n_query, args.num_segments, 3) +
                                                query_images.size()[-2:])
            query_targets = query_targets.reshape((n_way * n_query, -1))
            with torch.cuda.amp.autocast():
                if n_query > 20:
                    n_times = math.ceil(n_query / 20)
                    output = []
                    for i in range(n_times):
                        if i == n_times - 1:
                            images = query_images[i * 20 * n_way:]
                        else:
                            images = query_images[i * 20 * n_way: (i + 1) * 20 * n_way]
                        output.append(finetune_model(images))
                    output = torch.cat(output, 0)
                else:
                    output = finetune_model(query_images)
                if len(output) == 2:
                    acc0_1, acc0_5 = accuracy(output[0], query_targets, topk=(1, 5))
                    acc1_1, acc1_5 = accuracy(output[1], query_targets, topk=(1, 5))
                    alpha = args.alpha
                    acc1, acc5 = accuracy(output[0].softmax(1) * alpha + output[1].softmax(1) * (1 - alpha),
                                          query_targets, topk=(1, 5))
                else:
                    acc0_1, acc0_5 = accuracy(output, query_targets, topk=(1, 5))
                    acc1_1 = acc0_1
                    acc1 = acc0_1
                    acc1_5 = acc0_5
                    acc5 = acc0_5
            print('* Acc@1 {top1:.3f} Acc@5 {top5:.3f} '
                  'Acc0@1 {top01:.3f} Acc0@5 {top05:.3f} '
                  'Acc1@1 {top11:.3f} Acc1@5 {top15:.3f}'
                  .format(top1=acc1, top5=acc5,
                          top01=acc0_1, top05=acc0_5,
                          top11=acc1_1, top15=acc1_5))
            acc0_all.append(acc1.cpu())
            acc1_all.append(acc1_1.cpu())
    acc0_all = np.asarray(acc0_all)
    acc0_mean = np.mean(acc0_all)
    acc0_std = np.std(acc0_all)
    acc1_all = np.asarray(acc1_all)
    acc1_mean = np.mean(acc1_all)
    acc1_std = np.std(acc1_all)
    print(f'iter num: {len(test_loader)}, {acc0_mean} +- {acc0_std}')
    print(f'iter num: {len(test_loader)}, {acc1_mean} +- {acc1_std}')


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device)

    cudnn.benchmark = True

    test_datamgr = SetDataManager(args.data_root_val, 'test', input_size=args.input_size,
                                  n_way=args.n_way, n_support=args.n_support, n_query=args.n_query,
                                  num_segments=args.num_segments, n_eposide=args.n_eposide,
                                  args=args)
    test_loader = test_datamgr.get_data_loader(args.val_list_file, args)
    pretrain_model = create_model(
        args.pretrain_model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        dataset=test_datamgr,
        args=args)

    vis_backbone_path = getattr(args, 'vis_backbone_path',
                                osp.join(args.pretrain_cvlp_path, "checkpoint.pth"))
    pretrain_model.to(device)
    if vis_backbone_path.endswith('.pt'):
        checkpoint = {}
        checkpoint['model'] = torch.jit.load(
            vis_backbone_path, map_location=torch.device('cpu')).state_dict()
    else:
        checkpoint = torch.load(vis_backbone_path, map_location='cpu')

    if 'model' in checkpoint:
        msg = pretrain_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        msg = pretrain_model.load_state_dict(checkpoint, strict=False)
    print(msg)

    fewshot_test(test_loader, test_datamgr, pretrain_model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('fewshot script', parents=[get_args_parser()])
    args = parser.parse_args()
    args = utils.update_from_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
