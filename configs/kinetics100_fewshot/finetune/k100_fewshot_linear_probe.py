cfg = dict(
    pretrain_model='CVLP_r50',
    finetune_model='LGR_r50_no_init',
    desc_path='data/kinetics100_test',
    data_root_train='data/kinetics400/',
    data_root_val='data/kinetics400/',
    pretrained_clip='pretrained/RN50.pt',
    context_length=75,
    pretrain_cvlp_path='checkpoints/k100_base_pretrain_r50/',
    vis_backbone_path='checkpoints/k100_base_pretrain_r50/checkpoint.pth',

    use_mcloader=True,
    data_set='k100_support_query',
    dataset='k100_support_query',
    drop_last=True,
    index_bias=0,
    nb_classes=5,

    train_mode=False,
    train_list_file='data/kinetics100_test/k100_support_query_list.txt',
    val_list_file='data/kinetics100_test/k100_support_query_list.txt',

    epochs=50,
    batch_size=int(16),
    use_res=True,
    val_interval=10,
    save_interval=50,

    repeated_aug=False,
    mixup=0.,
    cutmix=0.,
    clip_ms=True,
    num_segments=16,
    new_length=1,
    is_video=True,
    select_num=50,

    io_backend='disk',
    only_video=False,
    num_classes=24,

    n_way=5,
    n_support=5,
    n_query=20,
    n_eposide=200,

    dropout=0.,
    emb_dropout=0.,
    consider_fusion_module=False,

    alpha=0.1,
    C=0.316,
    max_iter=1000,
    verbose=0
)