# VLG: General Video Recognition with Web Textual Knowledge

## Usage

First, install PyTorch 1.7.1+, torchvision 0.8.2+ and other required packages as follows:

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install mmcv==1.3.14
pip install decord
pip install git+https://github.com/ildoonet/pytorch-randaugment
```

## Data preparation

### Kinetics-Close/Kinetics-LT

Download the Kinetics videos from [here](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics).

Then download and extract the [wiki text](https://github.com/MCG-NJU/VLG/releases/tag/text-corpus) into the same directory. The directory tree of data is expected to be like this:

```
./data/kinetics400/
  videos_train/
    vid1.mp4
    ...
  videos_val/
    vid2.mp4
    ...
  wiki/
    desc_0.txt
    ...
  k400_LT_train_videos.txt
  k400_LT_val_videos.txt
  kinetics_video_train_list.txt
  kinetics_video_val_list.txt
  labels.txt
```

### Kinetics-Fewshot

We used the split from [CMN](https://github.com/ffmpbgrnn/CMN/tree/master/kinetics-100) for Kinetics-Fewshot.

Download and extract the [wiki text](https://github.com/MCG-NJU/VLG/releases/tag/text-corpus) into the same directory. The directory tree of data is expected to be like this:

```
./data/kinetics100_base
  wiki/
    desc_0.txt
    ...
  k100_base_train_list.txt
  labels.txt
./data/kinetics100_test
  wiki/
    desc_0.txt
    ...
  k100_support_query_list.txt
  labels.txt
```

### Kinetics-Fewshot-C-way

we used the split from [Efficient-Prompt](https://github.com/ju-chen/Efficient-Prompt) for Kinetics-Fewshot-C-way.

Download and extract the [wiki text](https://github.com/MCG-NJU/VLG/releases/tag/text-corpus) into the same directory. The directory tree of data is expected to be like this:

```
./data/kinetics400_fewshot_C
  wiki/
    desc_0.txt
    ...
  k400_fewshot_c_train_split_0.txt
  k400_fewshot_c_train_split_1.txt
  ...
  k400_fewshot_c_train_split_9.txt
  kinetics_video_val_list.txt
  labels.txt
```

### Kinetics-Openset

Download the split from [here](https://github.com/MCG-NJU/VLG/releases/tag/text-corpus) for Kinetics-Openset.

Then download and extract the [wiki text](https://github.com/MCG-NJU/VLG/releases/tag/text-corpus) into the same directory. The directory tree of data is expected to be like this:

```
./data/kinetics400_openset
  wiki/
    desc_0.txt
    ...
  k400_openset_train_list.txt
  k400_openset_val_list.txt
  labels.txt
```

## Evaluation

To evaluate VLG, you can run:

- Pre-training stage:

```
bash dist_train_arun.sh ${CONFIG_PATH} 8 --eval --eval-pretrain
```

- Fine-tuning stage:

```
bash dist_train_arun.sh ${CONFIG_PATH} 8 --eval
```

For fewshot cases, you can run:

```
bash dist_train_arun_fewshot.sh ${CONFIG_PATH} 8
```

For openset cases, you can run:

```
bash dist_train_arun_openset.sh ${CONFIG_PATH} 8 --test --dist-eval --eval
```

The `${CONFIG_PATH}` is the relative path of the corresponding configuration file in the `config` directory.

## Training

To train VLG on a single node with 8 GPUs for:

- Pre-training stage, run:

```
bash dist_train_arun.sh ${CONFIG_PATH} 8
```

- Fine-tuning stage:

  - First, select the salient sentences by running this:

    ```
    bash dist_train_arun.sh ${CONFIG_PATH} 8 --eval --select 
    ```

  - Then, running this:

    ```
    bash dist_train_arun.sh ${CONFIG_PATH} 8
    ```

The `${CONFIG_PATH}` is the relative path of the corresponding configuration file in the `config` directory.

## Pretrained Models:

Release soon.

## Citation

If you are interested in our work, please cite as follows:

```
@article{lin2022vlg,
  title={VLG: General Video Recognition with Web Textual Knowledge},
  author={Lin, Jintao and Liu, Zhaoyang and Wang, Wenhai and Wu, Wayne and Wang, Limin},
  journal={arXiv preprint arXiv:2212.01638},
  year={2022}
}
```

## Acknowledge

This repo contains modified codes from: [VL-LTR](https://github.com/ChangyaoTian/VL-LTR), [ActionCLIP](https://github.com/sallymmx/ActionCLIP), and [OpenMax](https://github.com/ma-xu/Open-Set-Recognition/tree/master/OSR/OpenMax).
