work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --cfg $work_path/test.yaml \
  DATA.PATH_TO_DATA_DIR /data/ZouYiShan/EpicData_rgb/label/test_8_2_V/ \
  DATA.PATH_PREFIX /data/ZouYiShan/EpicData_rgb/video/ \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 1 \
  NUM_GPUS 1 \
  UNIFORMER.DROP_DEPTH_RATE 0.1 \
  SOLVER.MAX_EPOCH 100 \
  SOLVER.BASE_LR 4e-4 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 1 \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH /data/ZouYiShan/Baselines/OpenEARsoft_cross/checkpoints/checkpoint_epoch_00100.pyth \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path
