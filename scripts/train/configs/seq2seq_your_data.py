from internnav.configs.model.base_encoders import (
    DepthEncoder,
    InstructionEncoder,
    ModelCfg,
    ProgressMonitor,
    RgbEncoder,
    Seq2Seq,
    StateEncoder,
    TextEncoder,
)
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import FilterFailure, IlCfg, Loss

# Seq2Seq config with your data - WITHOUT text_encoder (baseline test)
# This uses dummy instruction tokens and should NOT learn properly
seq2seq_your_data_model = ModelCfg(
    policy_name='Seq2Seq_Policy',
    max_step=200,
    len_traj_act=4,
    # Dummy instruction_encoder (meaningless tokens)
    instruction_encoder=InstructionEncoder(
        sensor_uuid='instruction',
        vocab_size=50265,  # Match CLIP-Long vocab for consistency
        use_pretrained_embeddings=False,
        embedding_file='',
        dataset_vocab='',
        fine_tune_embeddings=False,
        embedding_size=50,
        hidden_size=128,
        rnn_type='LSTM',
        final_state_only=True,
        bidirectional=False,
    ),
    # Text encoder - CLIP-Long for real semantic understanding
    text_encoder=TextEncoder(
        load_model=True,
        max_length=248,
        update_text_encoder=False,
        type='clip-long',
        model_name='clip-long',
        model_path='checkpoints/clip-long/longclip-B.pt',
        num_l_layers=6,
        hidden_size=512,
        vocab_size=50265,
        embedding_size=512,
        sot_token=49406,
        eot_token=49407,
        pad_token=0,
    ),
    rgb_encoder=RgbEncoder(cnn_type='TorchVisionResNet50', output_size=256, trainable=False),
    depth_encoder=DepthEncoder(
        cnn_type='VlnResnetDepthEncoder',
        output_size=128,
        backbone='resnet50',
        ddppo_checkpoint='checkpoints/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth',
        trainable=False,
    ),
    state_encoder=StateEncoder(
        hidden_size=512,
        rnn_type='GRU',
        num_recurrent_layers=1,
    ),
    progress_monitor=ProgressMonitor(
        use=True,
        alpha=1.0,
    ),
    seq2seq=Seq2Seq(
        use_prev_action=False,
    ),
)

seq2seq_your_data_exp_cfg = ExpCfg(
    name='seq2seq_your_data_train',
    model_name='seq2seq',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='checkpoints/%s/ckpts',
    checkpoint_folder='checkpoints/%s/ckpts',
    tensorboard_dir='checkpoints/%s/tensorboard',
    log_dir='checkpoints/%s/logs',
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        ckpt_to_load='',
        max_steps=195,
        sample=False,
        success_distance=3.0,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=55,
        save_interval_epochs=50, #5,
        batch_size=4,
        lr=1e-4,
        num_workers=8,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        use_iw=True,
        inflection_weight_coef=3.2,
        save_filter_frozen_weights=False,
        load_from_ckpt=False,
        ckpt_to_load='',
        load_from_pretrain=False,
        lmdb_map_size=1e12,
        dataset_r2r_root_dir='data/vln_pe/raw_data/r2r',
        dataset_3dgs_root_dir='',
        dataset_grutopia10_root_dir='/mnt/6t/dataset/my_vln_pe2/raw_data/vlnverse',
        lmdb_features_dir='grutopia',
        lerobot_features_dir='/mnt/6t/dataset/my_vln_pe2/traj_data/vlnverse',
        camera_name='pano_camera_0',
        report_to='tensorboard',
        ddp_find_unused_parameters=False,
        filter_failure=FilterFailure(
            use=True,
            min_rgb_nums=15,
        ),
        loss=Loss(
            alpha=0.0001,
            dist_scale=1,
        ),
    ),
    model=seq2seq_your_data_model,
)
