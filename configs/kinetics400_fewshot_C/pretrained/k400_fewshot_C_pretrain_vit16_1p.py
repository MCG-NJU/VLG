cfg = dict(
    model='CVLP_vit16_prompt',
    desc_path='data/kinetics400_fewshot_C',
    data_root_train='data/kinetics400/',
    data_root_val='data/kinetics400/videos_val/',
    pretrained_clip='pretrained/ViT-B-16.pt',
    context_length=75,
    pretrain_cvlp=True,
    loss_type="smoothCE",
    teacher_model="CVLP_vit16",

    use_mcloader=True,
    data_set='Kinetics',
    dataset='Kinetics',
    drop_last=True,
    index_bias=0,
    val_interval=5,

    weight_sample=True,
    use_sqrt_freq=True,
    train_mode=False,
    train_list_file='data/kinetics400_fewshot_C/k400_fewshot_c_train_split_1.txt',
    val_list_file='data/kinetics400_fewshot_C/kinetics_video_val_list.txt',
    use_res=True,

    lr=1.e-5,
    text_lr=1.e-5,
    fusion_lr=1.e-3,
    min_lr=0.,
    use_gpus=8,

    epochs=50,
    batch_size=int(16),

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
    num_classes=400,

    randaug_m=9,
    randaug_n=2,
    opt='adamw',
    sim_header='Transf',
    dropout=0.,
    emb_dropout=0.,
    place='blockres',
    joint=False,
    tsm=False,
    T=16,
    use_actionclip_solver=False,
    consider_fusion_module=True
)