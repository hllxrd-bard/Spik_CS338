CONFIG = {
    "dataset": "evcivil",
    "data_path": "/workingspace_aiclub/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/dataset",
    "output_dir": "./logs_evcivil_det_yolox_new_window60_1",

    "device": "cuda:0",
    "epochs": 20,
    "batch_size": 8,
    "workers": 2,

    "T": 16,
    "input_height": 256,
    "input_width": 256,
    "window_ms": 60.0,

    "num_classes": 2,
    "class_names": "class0,class1",

    "lr": 1e-4,
    "weight_decay": 1e-4,
    "amp": True,

    "max_train_sequences": None,
    "max_train_samples": 8000,
    "max_val_sequences": None,
    "max_val_samples": 1000,

    "print_freq": 100,
    "log_every_iters": 50,
    "save_every_iters": 1000,
    "keep_last_k_iters": 3,
    "resume": "/AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/spikformer/evcivil/logs_evcivil_det_yolox_new_window60/evcivil_spikformer_lif_yolox_T16_win60.0ms_256x256_ed256_d2_lr0.0001/checkpoint_latest.pth",

    "eval_metrics_every": 1,
    "disable_eval_metrics": False,

    "score_thr": 0.001,
    "metric_score_thrs": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    "nms_thr": 0.5,
    "iou_thr": 0.5,
    "max_det": 300,

    "disable_snn_metrics": False,
    "spike_thr": 0.0,
    "sops_spike_thr": 0.0,
    "sops_include_prefix": "backbone",
    "snn_layer_topk": 20,


    # Model config
    "model_in_channels": 2,
    "model_embed_dims": 256,
    "model_num_heads": 16,
    "model_depths": 2,
    "model_mlp_ratio": 4.0,
    "model_drop_path_rate": 0.1,

    # Spiking neuron config
    "neuron_type": "lif",          # "lif" hoặc "plif"
    "neuron_tau": 2.0,
    "neuron_v_threshold": 1.0,
    "neuron_attn_v_threshold": 0.5,
    "neuron_v_reset": 0.0,
    "neuron_detach_reset": True,
    "neuron_backend": "cupy",      # test PLIF có thể đổi thành "torch" trước
}