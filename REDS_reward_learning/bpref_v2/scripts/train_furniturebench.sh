export experiment_name='FurnitureBench-ARPDT'

ONLINE=True
SEED=${1}

# Env
ENV_NAME=${2}

# Dataset
DATA_PATH=${3}
NUM_DEMONSTRATIONS=${4}
WINDOW_SIZE=${5}

# --------------MODEL------------------
MODEL_TYPE="vit_base"
TRANSFER_TYPE="m3ae_vit_b16"
USE_ADAPTER=True
LAMBDA_RETURN_PRED=${6}

# --------------ARP------------------
USE_VL=${7}
VL_TYPE=${8}
RTG_KEY=${9}
VL_CHECKPOINT=${10}

# --------------TRAINING------------------
BATCH_SIZE=128
EPOCHS=100
TEST_EVERY_EPOCHS=25
LEARNING_RATE=5e-4
LR_SCHEDULE=cos

# --------------EVALUATION------------------
NUM_TEST_EPISODES=10

# --------------COMMENT------------------
COMMENT=${11}
NOTE="$COMMENT"
echo "note: $NOTE"

python3 -m bpref_v2.agents.bc_transformer.main_furniturebench \
    --is_tpu=False \
    --seed="$SEED" \
    --env_name="$ENV_NAME"\
    --data.data_dir="$DATA_PATH" \
    --data.num_demos="$NUM_DEMONSTRATIONS" \
    --data.window_size="$WINDOW_SIZE" \
    --window_size="$WINDOW_SIZE" \
    --data.use_bert_tokenizer=True \
    --data.rtg_key="$RTG_KEY" \
    --model.model_type="$MODEL_TYPE" \
    --model.transfer_type="$TRANSFER_TYPE" \
    --model.use_adapter="$USE_ADAPTER" \
    --model.lambda_return_pred="$LAMBDA_RETURN_PRED" \
    --val_every_epochs=$(($EPOCHS / 5)) \
    --test_every_epochs="$TEST_EVERY_EPOCHS" \
    --num_test_episodes="$NUM_TEST_EPISODES" \
    --batch_size="$BATCH_SIZE" \
    --weight_decay=5e-5 \
    --lr="$LEARNING_RATE" \
    --auto_scale_lr=False \
    --lr_schedule="$LR_SCHEDULE" \
    --warmup_epochs=10 \
    --momentum=0.9 \
    --clip_gradient=10.0 \
    --epochs="$EPOCHS" \
    --dataloader_n_workers=16 \
    --dataloader_shuffle=True \
    --log_all_worker=False \
    --logging.online="$ONLINE" \
    --logging.prefix='' \
    --logging.project="$experiment_name" \
    --logging.output_dir="/home/logdir/$experiment_name/${GAME}_${ENV_TYPE}" \
    --logging.random_delay=0.0 \
    --logging.notes="$NOTE" \
    --use_vl="$USE_VL" \
    --vl_type="$VL_TYPE" \
    --vl_checkpoint="$VL_CHECKPOINT" \
