<h1 align="left"> REDS reward learning
</h1>

This is the reward learning code for [REDS: Reward learning from Demonstration with Segmentations](https://arxiv.org/abs/2502.20630). For overall reward learning pipeline, and agent learning, please refer to [REDS_agent](https://github.com/csmile-1006/REDS_agent).

### 1. Install Dependencies

```
conda create -y -n reds python=3.10
conda activate reds

cd {path to the repository}
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install pre-commit
pre-commit install
```

### 2. Train Reward Model
If you want to train reward model for each phase separately, you can use the following commands.

#### Metaworld, RLBench
```python
# REDS (initial training, only with expert demonstrations)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m bpref_v2.reward_learning.train_reds \
    --comment={experiment_name} \
    --robot.data_dir={input_data_path} \
    --logging.output_dir={output_path} \
    --batch_size=32 \
    --model_type=REDS \
    --early_stop=False \
    --log_period=100 \
    --eval_period=1000 \
    --save_period=1000 \
    --train_steps=5000 \
    --eval_steps=10 \
    --robot.task_name={task_name} \
    --robot.num_demos={number of demos, to use all demos in the folder, use -1} \
    --robot.benchmark={metaworld|rlbench} \
    --env={metaworld|rlbench_{task_name}} \
    --robot.window_size=4 \
    --robot.skip_frame=1 \
    --reds.lambda_supcon=1.0 \
    --reds.lambda_epic 1.0 \
    --reds.transfer_type=clip_vit_b16 \
    --reds.embd_dim=512 \
    --reds.output_embd_dim=512 \
    --augmentations="crop|jitter" \
    --robot.output_type raw \
    --logging.online=True \
    --use_failure=False \
    --reds.epic_on_neg_batch=False \
    --reds.supcon_on_neg_batch=False \
    --robot.pearson_size=8

# REDS (iterative training with expert + suboptimal demonstrations)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m bpref_v2.reward_learning.train_reds \
    --comment={experiment_name} \
    --robot.data_dir={input_data_path} \
    --logging.output_dir={output_path} \
    --batch_size=32 \
    --model_type=REDS \
    --early_stop=False \
    --log_period=100 \
    --eval_period=1000 \
    --save_period=1000 \
    --train_steps=5000 \
    --eval_steps=10 \
    --robot.task_name={task_name} \
    --robot.num_demos={number of demos, to use all demos in the folder, use -1} \
    --num_failure_demos={number of failure demos, to use all failure demos in the folder, use -1} \
    --robot.env_type={metaworld|rlbench} \
    --env={metaworld|rlbench_{task_name}} \
    --robot.window_size=4 \
    --robot.skip_frame=1 \
    --reds.lambda_supcon=1.0 \
    --reds.lambda_epic 1.0 \
    --reds.transfer_type=clip_vit_b16 \
    --reds.embd_dim=512 \
    --reds.output_embd_dim=512 \
    --augmentations="crop|jitter" \
    --robot.output_type raw \
    --logging.online=True \
    --use_failure=True \
    --reds.epic_on_neg_batch=True \
    --reds.supcon_on_neg_batch=True \
    --robot.pearson_size=8
```

#### FurnitureBench
```python
# REDS (initial training, only with expert demonstrations)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m bpref_v2.reward_learning.train_reds \
    --comment={experiment_name} \
    --furniturebench.data_dir={input_data_path} \
    --logging.output_dir={output_path} \
    --batch_size=32 \
    --model_type=REDS \
    --early_stop=False \
    --log_period=100 \
    --eval_period=1000 \
    --save_period=1000 \
    --train_steps=5000 \
    --eval_steps=10 \
    --furniturebench.task_name=one_leg \
    --furniturebench.num_demos={number of demos, to use all demos in the folder, use -1} \
    --furniturebench.env_type=furniturebench \
    --env=furniturebench-one_leg \
    --furniturebench.window_size=4 \
    --furniturebench.skip_frame=1 \
    --reds.lambda_supcon=1.0 \
    --reds.lambda_epic 1.0 \
    --reds.transfer_type=clip_vit_b16 \
    --reds.embd_dim=512 \
    --reds.output_embd_dim=512 \
    --augmentations="crop|jitter" \
    --furniturebench.output_type raw \
    --logging.online=True \
    --use_failure=False \
    --reds.epic_on_neg_batch=False \
    --reds.supcon_on_neg_batch=False \
    --furniturebench.pearson_size=8

# REDS (iterative training with expert + suboptimal demonstrations)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m bpref_v2.reward_learning.train_reds \
    --comment={experiment_name} \
    --furniturebench.data_dir={input_data_path} \
    --logging.output_dir={output_path} \
    --batch_size=32 \
    --model_type=REDS \
    --early_stop=False \
    --log_period=100 \
    --eval_period=1000 \
    --save_period=1000 \
    --train_steps=5000 \
    --eval_steps=10 \
    --furniturebench.task_name=one_leg \
    --furniturebench.num_demos={number of demos, to use all demos in the folder, use -1} \
    --num_failure_demos={number of failure demos, to use all failure demos in the folder, use -1} \
    --furniturebench.env_type=furniturebench \
    --env=furniturebench-one_leg \
    --furniturebench.window_size=4 \
    --furniturebench.skip_frame=1 \
    --reds.lambda_supcon=1.0 \
    --reds.lambda_epic 1.0 \
    --reds.transfer_type=clip_vit_b16 \
    --reds.embd_dim=512 \
    --reds.output_embd_dim=512 \
    --augmentations="crop|jitter" \
    --furniturebench.output_type raw \
    --logging.online=True \
    --use_failure=True \
    --reds.epic_on_neg_batch=True \
    --reds.supcon_on_neg_batch=True \
    --furniturebench.pearson_size=8
```

## BibTeX
```
@inproceedings{kim2025subtask,
  title={Subtask-Aware Visual Reward Learning from Segmented Demonstrations},
  author={Kim, Changyeon and Heo, Minho and Lee, Doohyun and Shin, Jinwoo and Lee, Honglak and Lim, Joseph J. and Lee, Kimin},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
}
```


## Acknowledgments

Our code is based on the implementation of [PreferenceTransformer](https://github.com/csmile-1006/PreferenceTransformer).
