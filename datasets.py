from mcloader.transforms_ss import *
from RandAugment import RandAugment

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mcloader import ClassificationDataset


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


CLIP_DEFAULT_MEAN = (0.4815, 0.4578, 0.4082)
CLIP_DEFAULT_STD = (0.2686, 0.2613, 0.2758)


def build_dataset(split, args):
    assert split in ['train', 'val', 'test']
    is_train = split == "train"
    if is_train:
        list_file = args.train_list_file
    else:
        list_file = args.val_list_file

    transform = build_transform(is_train, args, split)
    if is_train and args.randaug_n > 0:
        transform = randAugment(transform, args)

    assert args.data_set in ['UCF101', 'HMDB51', 'Kinetics',
                             "Kinetics100_base", 'Kinetics100_test', 'k100_support_query',
                             'kinetics400_openset']

    if args.data_set == "UCF101":
        nb_classes = 101
    elif args.data_set == "HMDB51":
        nb_classes = 51
    elif args.data_set == "Kinetics":
        nb_classes = 400
    elif args.data_set == 'Kinetics100_base':
        nb_classes = 64
    elif args.data_set == 'Kinetics100_test':
        nb_classes = 24
    elif args.data_set == 'k100_support_query':
        nb_classes = 5
    elif args.data_set == 'kinetics400_openset':
        nb_classes = 250
    else:
        nb_classes = 200

    dataset = ClassificationDataset(
        args.data_root_train if is_train else args.data_root_val,
        list_file,
        split=split,
        nb_classes=nb_classes,
        desc_path=args.desc_path,
        context_length=args.context_length,
        pipeline=transform,
        transform=transform,
        select=args.select,
        num_segments=args.num_segments,
        new_length=args.new_length,
        dataset=args.dataset,
        is_video=args.is_video,
        select_num=args.select_num,
        index_bias=args.index_bias,
        test_mode=(not is_train),
        io_backend=args.io_backend,
        only_video=args.only_video,
        dense_sample=args.dense_sample,
        num_clips=args.num_clips,
        twice_sample=args.twice_sample,
        naive_txt=args.naive_txt)
    nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args, split):
    DEFAULT_MEAN = CLIP_DEFAULT_MEAN if args.clip_ms else IMAGENET_DEFAULT_MEAN
    DEFAULT_STD = CLIP_DEFAULT_STD if args.clip_ms else IMAGENET_DEFAULT_STD
    scale_size = args.input_size * args.scale_size // args.input_size
    if is_train:
        unique = torchvision.transforms.Compose([GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66]),
                                                 GroupRandomHorizontalFlip(is_sth='some' in args.data_set),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0.0),
                                                 GroupSolarization(p=0.0)])
    else:
        if args.test_crops == 1 or split == 'val':
            unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                     GroupCenterCrop(args.input_size)])
        elif args.test_crops == 3:
            unique = torchvision.transforms.Compose([GroupFullResSample(args.input_size, scale_size, flip=False)])
        elif args.test_crops == 5:
            unique = torchvision.transforms.Compose([GroupOverSample(args.input_size, scale_size, flip=False)])
        elif args.test_crops == 10:
            unique = torchvision.transforms.Compose([GroupOverSample(args.input_size, scale_size)])
        else:
            raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(DEFAULT_MEAN,
                                                            DEFAULT_STD)])
    return torchvision.transforms.Compose([unique, common])


def randAugment(transform_train, config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.randaug_n, config.randaug_m)))
    return transform_train