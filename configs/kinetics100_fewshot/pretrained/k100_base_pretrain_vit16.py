cfg = dict(
    model='CVLP_vit16',
    desc_path='data/kinetics100_base',
    data_root_train='data/kinetics400/',
    data_root_val='data/kinetics400/',
    pretrained_clip='pretrained/ViT-B-16.pt',
    context_length=75,
    pretrain_cvlp=True,
    loss_type="smoothCE",

    use_mcloader=True,
    data_set='Kinetics100_base',
    dataset='Kinetics100_base',
    drop_last=True,
    index_bias=0,

    weight_sample=True,
    use_sqrt_freq=True,
    train_mode=False,
    train_list_file='data/kinetics100_base/k100_base_train_list.txt',
    val_list_file='data/kinetics100_base/k100_base_train_list.txt',

    lr=1.e-5,
    text_lr=1.e-5,
    min_lr=0.,
    use_gpus=8,

    epochs=50,
    batch_size=int(32),

    repeated_aug=False,
    mixup=0.,
    cutmix=0.,
    clip_ms=True,
    distillation_beta=0.5,
    distillation_type='logits',
    num_segments=16,
    new_length=1,
    is_video=True,
    select_num=50,

    eval_pretrain=True,
    io_backend='disk',
    only_video=False,
    broadcast_bn_buffer=True,
    find_unused_parameters=False,
    num_classes=64,

    opt='adamw',
    consider_fusion_module=False
)