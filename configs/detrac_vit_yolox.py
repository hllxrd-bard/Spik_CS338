CONFIG = {
    "dataset": "detrac",
    "data_path": "/workingspace_aiclub/WorkingSpace/Personal/chinhnm/HLLXRD/ua_detrac/dataset/datasets/bratjay/ua-detrac-orig/versions/2",
    "output_dir": "./logs_detrac_det_vit_yolox_2",

    "device": "cuda:0",
    "epochs": 20,
    "batch_size": 8,
    "workers": 4,

    "T": 16,
    "input_height": 256,
    "input_width": 256,
    "window_ms": 30.0,

    "representation": "grayscale_dup",
    "frame_stride": 1,

    "num_classes": 3,
    "class_names": "car,van,bus",

    "lr": 1e-4,
    "weight_decay": 1e-4,
    "amp": True,

    "max_train_sequences": None,
    "max_train_samples": 40000,
    "max_val_sequences": None,
    "max_val_samples": 2000,

    "print_freq": 100,
    "log_every_iters": 50,
    "save_every_iters": 2000,
    "keep_last_k_iters": 3,
    "resume": "/AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/spikformer/evcivil/logs_detrac_det_vit_yolox/detrac_vit_yolox_T16_256x256_lr0.0001/checkpoint_latest.pth",

    "eval_metrics_every": 1,
    "disable_eval_metrics": False,

    "score_thr": 0.001,
    "metric_score_thrs": [0.001, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
    "nms_thr": 0.5,
    "iou_thr": 0.5,
    "max_det": 300,

    "disable_snn_metrics": True,
    "spike_thr": 0.0,
    "sops_spike_thr": 0.0,
    "sops_include_prefix": "backbone",
    "snn_layer_topk": 20,
}