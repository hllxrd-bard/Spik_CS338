CONFIG = {
    # Dataset
    "dataset": "detrac",
    "data_path": "/AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/spikformer/ua_detrac/dataset/datasets/bratjay/ua-detrac-orig/versions/2",
    "output_dir": "./logs_detrac_det_yolox",

    # Device / training
    "device": "cuda:0",
    "epochs": 20,
    "batch_size": 4,
    "workers": 4,

    # Input
    "T": 16,
    "input_height": 256,
    "input_width": 256,
    "window_ms": 30.0,

    # DETRAC-specific
    "representation": "grayscale_dup",
    "frame_stride": 1,

    # Classes
    "num_classes": 3,
    "class_names": "car,van,bus",

    # Optimizer
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "amp": True,

    # Debug subset
    "max_train_samples": 200,
    "max_val_samples": 200,
    "max_train_sequences": 1,
    "max_val_sequences": 1,

    # Logging / checkpoint
    "print_freq": 100,
    "log_every_iters": 50,
    "save_every_iters": 1000,
    "keep_last_k_iters": 3,
    "resume": "",

    # Detection eval metrics
    "eval_metrics_every": 1,
    "disable_eval_metrics": False,
    "score_thr": 0.001,
    "metric_score_thrs": "0.001,0.01,0.05,0.1,0.2,0.3,0.5",
    "nms_thr": 0.5,
    "iou_thr": 0.5,
    "max_det": 300,

    # SNN metrics
    "disable_snn_metrics": False,
    "spike_thr": 0.0,
    "sops_spike_thr": 0.0,
    "sops_include_prefix": "backbone",
    "snn_layer_topk": 20,
}