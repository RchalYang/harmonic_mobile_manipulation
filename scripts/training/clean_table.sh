
WANDB_PROJECT=harmonic_mm

RUN_NAME=clean_table

OUTPUT_DIR=results/clean_table

SEED=1

PYTHONPATH=$PYTHONPATH:$PWD python training/main.py \
  experiment=training/experiments_dino_single/cleaning_table_multimodalv2_dinovitgru.py \
  task.name=CleaningTableTask \
  task_config=cleaning_table \
  agent=stretch \
  target_object_types=cleaning_table \
  wandb.project=$WANDB_PROJECT \
  machine.num_val_processes=2 \
  machine.num_train_processes=20 \
  machine.num_test_processes=0 \
  ai2thor.platform=CloudRendering \
  procthor.p_randomize_materials=1.0 \
  wandb.name=$RUN_NAME \
  model.model_type_name=VitMultiModalPrevActV2NCameraActorCritic \
  sensor.enable_history=False \
  training.entropy_coef=0.0025 \
  training.lr=0.00005 \
  training.long_horizon=False \
  model.no_prev_act=True \
  training.small_batch=False \
  mdp.reward.opening_door.train.energy_penalty=-0.05 \
  mdp.reward.opening_door.train.open_initial_reward=0.0 \
  mdp.reward.opening_door.train.open_section_reward=80.0 \
  output_dir=$OUTPUT_DIR \
  seed=$SEED
