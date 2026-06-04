# configs/evcivil_vit_yolox.py

CONFIG = {
    "dataset": "evcivil",
    "data_path": "/workingspace_aiclub/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/dataset",
    "output_dir": "./logs_evcivil_det_vit_yolox_new",

    "device": "cuda:0",
    "epochs": 20,
    "batch_size": 8,
    "workers": 2,

    "T": 16,
    "input_height": 256,
    "input_width": 256,
    "window_ms": 30.0,

    "num_classes": 2,
    "class_names": "crack,spalling",

    "lr": 1e-4,
    "weight_decay": 1e-4,
    "amp": True,

    "max_train_sequences": None,
    "max_train_samples": 8000,
    "max_val_sequences": None,
    "max_val_samples": 1000,

    "print_freq": 100,
    "log_every_iters": 50,
    "save_every_iters": 200,
    "keep_last_k_iters": 3,
    "resume": "",

    "eval_metrics_every": 1,
    "disable_eval_metrics": False,

    "score_thr": 0.001,
    "metric_score_thrs": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    "nms_thr": 0.5,
    "iou_thr": 0.5,
    "max_det": 300,

    "disable_snn_metrics": True,
    "spike_thr": 0.0,
    "sops_spike_thr": 0.0,
    "sops_include_prefix": "backbone",
    "snn_layer_topk": 20,

    "vit_embed_dims": 256,
    "vit_depths": [1, 1, 1],
    "vit_heads": 8,
    "vit_mlp_ratio": 4.0,
    "vit_dropout": 0.0,
}