# 8 GPU
cfg = dict(
    model='LGR_vit16_prompt',
    desc_path='data/kinetics400',
    data_root_train='data/kinetics400/videos_train/',
    data_root_val='data/kinetics400/videos_val/',
    pretrained_clip='pretrained/ViT-B-16.pt',
    context_length=75,
    pretrain_cvlp=False,
    pretrain_cvlp_path='checkpoints/k400_pretrain_vit16',
    loss_type="CE",
    two_branch=True,

    use_mcloader=True,
    data_set='Kinetics',
    dataset='Kinetics',
    drop_last=True,
    index_bias=0,

    weight_sample=True,
    use_sqrt_freq=True,
    train_list_file='data/kinetics400/kinetics_video_train_list.txt',
    val_list_file='data/kinetics400/kinetics_video_val_list.txt',

    lr=1e-3,
    min_lr=0,
    warmup_epochs=0,
    text_lr=1e-5,

    epochs=50,
    batch_size=int(128),
    is_video=True,
    num_segments=8,
    new_length=1,
    select_num=50,

    repeated_aug=False,
    clip_ms=True,
    io_backend='disk',
    only_video=False,
    find_unused_parameters=False,
    broadcast_bn_buffer=True,
    num_classes=400,

    opt='adamw',
    sim_header='Transf',
    dropout=0.,
    emb_dropout=0.,
    place='blockres',
    joint=False,
    tsm=False,
    T=8,
    consider_fusion_module=False
)