from mcloader.transforms_ss import *
import os
import os.path as osp
import pickle
from mmcv.fileio import FileClient
import torch.distributed as dist
from mcloader.preprocess import SentPreProcessor
import decord
import io
from numpy.random import randint

CLIP_DEFAULT_MEAN = (0.4815, 0.4578, 0.4082)
CLIP_DEFAULT_STD = (0.2686, 0.2613, 0.2758)

DEFAULT_MEAN = CLIP_DEFAULT_MEAN
DEFAULT_STD = CLIP_DEFAULT_STD

identity = lambda x: x


class FrameRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class SubVdieoDataset:
    def __init__(self, root, sub_meta, cl, split='test', transform=None,
                 target_transform=identity,
                 random_select=False, num_segments=None, args=None):
        self.root = root
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.random_select = random_select
        self.num_segments = num_segments
        self.args = args

        is_train = split == "train"
        self.split = split
        self.test_mode = (not is_train)
        self.new_length = getattr(args, 'new_length', 1)
        if self.test_mode:
            self.random_shift = False
        else:
            self.random_shift = True
        self.seg_length = getattr(args, 'seg_length', 1)
        self.io_backend = getattr(args, 'io_backend', 'disk')
        self.dense_sample = getattr(args, 'dense_sample', False)
        self.twice_sample = getattr(args, 'twice_sample', False)
        assert not (self.dense_sample and self.twice_sample)
        self.num_sample_position = getattr(args, 'num_sample_position', 64)
        self.num_clips = getattr(args, 'num_clips', 1)
        self.naive_txt = getattr(args, 'naive_txt', False)
        self.loop = False
        self.index_bias = getattr(args, 'index_bias', 0)
        self.nb_classes = getattr(args, 'nb_classes', 5)
        self.context_length = getattr(args, 'context_length', 0)
        self.select = getattr(args, 'select', False)
        self.is_video = getattr(args, 'is_video', True)
        self.select_num = getattr(args, 'select_num', 50)
        self.num_threads = getattr(args, 'num_threads', 1)
        self.only_video = getattr(args, 'only_video', False)

        if self.is_video:
            self.index_bias = 0

        self.initialized = False
        if self.io_backend == 'mc':
            self.mc_cfg = dict(
                server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
                client_cfg='/mnt/lustre/share/memcached_client/client.conf',
                sys_path='/mnt/lustre/share/pymc/py3')
            self.file_client = FileClient('memcached', **self.mc_cfg)
        if self.io_backend == 'disk':
            self.file_client = FileClient(self.io_backend)
        if self.io_backend == 'petrel':
            self.file_client = None

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    def _sample_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - self.num_sample_position)
            t_stride = self.num_sample_position // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + self.index_bias
        else:
            if record.num_frames <= self.total_length:
                if self.loop:
                    return np.mod(np.arange(
                        self.total_length) + randint(record.num_frames // 2),
                                  record.num_frames) + self.index_bias
                offsets = np.concatenate((
                    np.arange(record.num_frames),
                    randint(record.num_frames,
                            size=self.total_length - record.num_frames)))
                return np.sort(offsets) + self.index_bias
            offsets = list()
            ticks = [i * record.num_frames // self.num_segments
                     for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += randint(tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.dense_sample:
            assert self.split == 'test' or (self.split == 'train' and self.select == True)
            sample_pos = max(1, 1 + record.num_frames - self.num_sample_position)
            t_stride = self.num_sample_position // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=self.num_clips, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + self.index_bias
        elif self.twice_sample:
            tick = (record.num_frames) / float(self.num_segments)
            coeffs = np.arange(self.num_clips) / self.num_clips
            offsets = []
            for coeff in coeffs:
                offsets = offsets + [int(tick * coeff + tick * x) for x in range(self.num_segments)]
            offsets = np.array(offsets)
            return offsets + self.index_bias
        else:
            if self.num_segments == 1:
                return np.array([record.num_frames // 2], dtype=np.int) + self.index_bias

            if record.num_frames <= self.total_length:
                if self.loop:
                    return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
                return np.array([i * record.num_frames // self.total_length
                                 for i in range(self.total_length)], dtype=np.int) + self.index_bias
            offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
            return np.array([i * record.num_frames / self.num_segments + offset + j
                             for i in range(self.num_segments)
                             for j in range(self.seg_length)], dtype=np.int) + self.index_bias

    def get_video(self, record, indices):
        images = record.video.get_batch(indices).asnumpy()
        del record.video
        img_list = []
        for img in images:
            img_list.append(Image.fromarray(np.uint8(img)))
        process_data = self.transform(img_list)
        return process_data, record.label

    def __getitem__(self, i):
        assert len(self.sub_meta[i]) == 2
        full_path = self.sub_meta[i][0]
        label = self.sub_meta[i][1]
        assert self.is_video
        assert label == self.cl
        assert self.split == 'test'
        record = VideoRecord([full_path, label])

        if self.io_backend == 'disk':
            setattr(record, 'video', decord.VideoReader(osp.join(self.root, record.path), num_threads=self.num_threads))
        elif self.io_backend == 'mc':
            vid_path = osp.join(self.root, record.path)
            file_obj = io.BytesIO(self.file_client.get(vid_path))
            setattr(record, 'video', decord.VideoReader(file_obj, num_threads=self.num_threads))
        elif self.io_backend == 'petrel':
            vid_path = osp.join(self.root, record.path)
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend)
            file_obj = io.BytesIO(self.file_client.get(vid_path))
            setattr(record, 'video', decord.VideoReader(file_obj, num_threads=self.num_threads))
        setattr(record, 'num_frames', len(record.video))
        if self.split == 'train' and self.select:
            segment_indices = self._get_val_indices(record)
        elif self.split == 'train':
            segment_indices = self._sample_indices(record)
        elif self.split == 'val' and self.dense_sample:
            segment_indices = self._sample_indices(record)
        elif self.split == 'val':
            segment_indices = self._get_val_indices(record)
        elif self.split == 'test':
            segment_indices = self._get_val_indices(record)
        return self.get_video(record, segment_indices)

    def __len__(self):
        return len(self.sub_meta)


class TransformLoader:
    def __init__(self, scale_size=256, input_size=224, args=None):
        self.scale_size = scale_size
        self.input_size = input_size
        self.args = args

    def get_composed_transform(self, args):
        test_crops = args.test_crops
        num_clips = args.num_clips

        if test_crops == 1:
            unique = torchvision.transforms.Compose([GroupScale(self.scale_size),
                                                     GroupCenterCrop(self.input_size)])
        elif test_crops == 3:
            unique = torchvision.transforms.Compose([GroupFullResSample(self.input_size, self.scale_size, flip=False)])
        elif test_crops == 5:
            unique = torchvision.transforms.Compose([GroupOverSample(self.input_size, self.scale_size, flip=False)])
        elif test_crops == 10:
            unique = torchvision.transforms.Compose([GroupOverSample(self.input_size, self.scale_size)])
        else:
            raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(test_crops))

        common = torchvision.transforms.Compose([Stack(roll=False),
                                                 ToTorchFormatTensor(div=True),
                                                 GroupNormalize(DEFAULT_MEAN,
                                                                DEFAULT_STD)])
        return torchvision.transforms.Compose([unique, common])


class SetDataset:
    def __init__(self, root, data_file, batch_size, transform,
                 random_select=False, num_segments=8,
                 args=None, split='test'):
        self.split = split
        self.io_backend = args.io_backend
        if self.io_backend != 'petrel':
            self.root = os.path.realpath(root)
        else:
            self.root = root
        self.video_list = [x.strip().split(' ') for x in open(data_file)]
        self.cl_list = np.zeros(len(self.video_list), dtype=int)
        for i in range(len(self.video_list)):
            self.cl_list[i] = int(self.video_list[i][1])  # here change
        self.cl_list = np.unique(self.cl_list).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        for x in range(len(self.video_list)):
            video_path = self.video_list[x][0]
            label = int(self.video_list[x][1])
            self.sub_meta[label].append([video_path, label])

        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size, shuffle=True,
                                      num_workers=0, pin_memory=False)
        for cl in self.cl_list:
            sub_dataset = SubVdieoDataset(self.root, self.sub_meta[cl], cl, transform=transform,
                                          random_select=random_select, num_segments=num_segments,
                                          args=args, split=split)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset,
                                                                   **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


def get_naive_tokens(dataset, desc_path, context_length):
    print('Use naive description')
    cache_root = 'cached'
    cache_path = osp.join(cache_root, '%s_naive_desc_text_sent.pkl' % dataset)
    clip_token_path = osp.join(cache_root, '%s_naive_text_tokens.pkl' % dataset)
    dist.barrier()
    if osp.exists(clip_token_path):
        with open(clip_token_path, 'rb') as f:
            text_tokens = pickle.load(f)
        return text_tokens

    dist.barrier()
    preprocessor = SentPreProcessor(root=desc_path, dataset=dataset)
    dist.barrier()
    if not osp.exists(cache_path):
        dist.barrier()
        os.makedirs(cache_root, exist_ok=True)
        texts = preprocessor.get_naive_text()
        texts = preprocessor.split_sent(texts)
        with open(cache_path, 'wb') as f:
            pickle.dump(texts, f)
    else:
        with open(cache_path, 'rb') as f:
            texts = pickle.load(f)
    dist.barrier()
    text_tokens = preprocessor.tokenize(texts, context_length=context_length)
    with open(clip_token_path, 'wb') as f:
        pickle.dump(text_tokens, f)
    return text_tokens


def get_sentence_tokens(dataset: str, desc_path, context_length):
    print('using clip text tokens splitted by sentence')
    cache_root = 'cached'
    cache_path = osp.join(cache_root, '%s_desc_text_sent.pkl' % dataset)
    clip_token_path = osp.join(cache_root, '%s_text_tokens.pkl' % dataset)
    dist.barrier()
    if osp.exists(clip_token_path):
        with open(clip_token_path, 'rb') as f:
            text_tokens = pickle.load(f)
        return text_tokens

    dist.barrier()
    preprocessor = SentPreProcessor(root=desc_path, dataset=dataset)
    dist.barrier()
    if not osp.exists(cache_path):
        dist.barrier()
        os.makedirs(cache_root, exist_ok=True)
        texts = preprocessor.get_clip_text()
        texts = preprocessor.split_sent(texts)
        with open(cache_path, 'wb') as f:
            pickle.dump(texts, f)
    else:
        with open(cache_path, 'rb') as f:
            texts = pickle.load(f)
    dist.barrier()
    text_tokens = preprocessor.tokenize(texts, context_length=context_length)
    with open(clip_token_path, 'wb') as f:
        pickle.dump(text_tokens, f)
    return text_tokens


class SetDataManager:
    def __init__(self, root, split, input_size=224, n_way=5,
                 n_support=5, n_query=20, num_segments=8,
                 n_eposide=200, args=None, dataset='k100_support_query'):
        super(SetDataManager, self).__init__()
        self.root = root
        self.split = split
        self.input_size = input_size
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.num_segments = num_segments
        self.n_eposide = n_eposide
        self.args = args

        self.batch_size = n_support + n_query
        self.trans_loader = TransformLoader(scale_size=args.scale_size, input_size=self.input_size,
                                            args=args)

        self.naive_txt = getattr(args, 'naive_txt', False)
        self.dataset = dataset
        self.desc_path = args.desc_path
        self.context_length = args.context_length

        if self.naive_txt:
            self.full_text_tokens = get_naive_tokens(dataset, self.desc_path, self.context_length)
            self.full_end_idxs = [len(sents) for sents in self.full_text_tokens]
        else:
            self.full_text_tokens = get_sentence_tokens(dataset, self.desc_path, self.context_length)
            self.full_end_idxs = [len(sents) for sents in self.full_text_tokens]

        self.full_classes_name = self.get_classes_name()

    def get_classes_name(self):
        with open(os.path.join(self.desc_path, "labels.txt"), "r") as rf:
            data = rf.readlines()
        _lines = [l.split() for l in data]
        categories = []
        for id, l in enumerate(_lines):
            name = '_'.join(l)
            name = name.replace("_", ' ')
            categories.append(name)
        return categories

    def get_data_loader(self, list_file, args):
        random_select = self.split == 'train'
        transform = self.trans_loader.get_composed_transform(args)
        dataset = SetDataset(self.root, list_file, self.batch_size, transform,
                             random_select=random_select, num_segments=self.num_segments,
                             args=args, split=self.split)  # video
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


class EpisodicBatchSampler:
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
