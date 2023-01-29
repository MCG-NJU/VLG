import io
import os
import os.path as osp
import pickle

from torch.utils.data import Dataset

from .preprocess import SentPreProcessor
from numpy.random import randint
import decord

import numpy as np
import torch
from PIL import Image
import torch.distributed as dist
from mmcv.fileio import FileClient


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return


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


class ClassificationDataset(Dataset):

    def __init__(self, root, list_file, labels_file=None, num_segments=8, new_length=1,
                 image_tmpl='img_{:05d}.jpg', random_shift=True, test_mode=False, index_bias=1,
                 dataset='Kinetics', split='train', nb_classes=400, transform=None,
                 desc_path='', context_length=0, pipeline=None, select=False, is_video=True,
                 select_num=50, num_threads=1, io_backend='disk', only_video=False, dense_sample=False,
                 num_sample_position=64, num_clips=10, twice_sample=False, naive_txt=False):

        assert dataset in ['UCF101', 'HMDB51', 'Kinetics', 'Kinetics100_base',
                           'Kinetics100_test', 'k100_support_query', 'kinetics400_openset']
        if io_backend != 'petrel':
            self.root = os.path.realpath(root)
        else:
            self.root = root
        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.io_backend = io_backend
        self.dense_sample = dense_sample
        self.twice_sample = twice_sample
        assert not (dense_sample and twice_sample)
        print(f'dense_sample: {self.dense_sample}, twice_sample: {self.twice_sample}, split: {split}')
        self.num_sample_position = num_sample_position
        self.num_clips = num_clips
        self.naive_txt = naive_txt

        if self.test_mode:
            self.random_shift = False
        else:
            self.random_shift = True
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.dataset = dataset
        self.split = split
        self.nb_classes = nb_classes
        self.desc_path = desc_path
        self.context_length = context_length
        self.pipeline = pipeline
        self.select = select
        self.is_video = is_video
        self.select_num = select_num
        self.num_threads = num_threads
        self.only_video = only_video

        if not self.only_video:
            if self.naive_txt:
                self.text_tokens = get_naive_tokens(dataset, desc_path, context_length)
                self.end_idxs = [len(sents) for sents in self.text_tokens]
            else:
                self.text_tokens = get_sentence_tokens(dataset, desc_path, context_length)
                self.end_idxs = [len(sents) for sents in self.text_tokens]

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        if self.is_video:
            self.index_bias = 0

        self._parse_list()
        self.initialized = False
        self.targets = self.labels
        self.classes_name = self.get_classes_name()
        if self.io_backend == 'mc':
            self.mc_cfg = dict(
                server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
                client_cfg='/mnt/lustre/share/memcached_client/client.conf',
                sys_path='/mnt/lustre/share/pymc/py3')
        if self.io_backend == 'disk':
            self.file_client = FileClient(self.io_backend)
        if self.io_backend == 'petrel':
            self.file_client = None
            
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

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    def _parse_list(self):
        if self.is_video:
            self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
            self.labels = [int(x.strip().split(' ')[-1]) for x in open(self.list_file)]
        else:
            self.video_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_file)]
            self.labels = [int(x.strip().split(' ')[-1]) for x in open(self.list_file)]
        if self.select:
            n_labels = []
            video_list = []
            cls_cnt_dict = {}
            for video in self.video_list:
                label = video.label
                if label not in cls_cnt_dict:
                    cls_cnt_dict[label] = 0
                cls_cnt_dict[label] += 1
                if cls_cnt_dict[label] > self.select_num:
                    continue
                video_list.append(video)
                n_labels.append(label)
            self.video_list = video_list
            self.labels = n_labels

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

    def _load_image(self, directory, idx):
        img_path = os.path.join(self.root, directory, self.image_tmpl.format(idx))
        if self.io_backend == 'disk':
            return [Image.open(img_path).convert('RGB')]
        else:
            img_bytes = self.file_client.get(img_path)
            with io.BytesIO(img_bytes) as buff:
                cur_frame = Image.open(buff).convert('RGB')
            return [cur_frame]

    def get(self, record, indices):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(os.path.join(self.root, record.path, p)))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    def get_length(self):
        return len(self.video_list)

    def get_video(self, record, indices):
        images = record.video.get_batch(indices).asnumpy()
        del record.video
        img_list = []
        for img in images:
            img_list.append(Image.fromarray(np.uint8(img)))
        process_data = self.transform(img_list)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.is_video:
            if self.split == 'train':
                segment_indices = self._sample_indices(record)
            elif self.split == 'val' and self.dense_sample:
                segment_indices = self._sample_indices(record)
            elif self.split == 'val':
                segment_indices = self._get_val_indices(record)
            elif self.split == 'test':
                segment_indices = self._get_val_indices(record)

            return self.get(record, segment_indices)
        else:
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
            if hasattr(record, 'video'):
                setattr(record, 'num_frames', len(record.video))
            else:
                print(vid_path, flush=True)
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
