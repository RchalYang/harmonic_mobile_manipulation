
WANDB_PROJECT=harmonic_mm

RUN_NAME=open_fridge 

OUTPUT_DIR=results/open_fridge

SEED=1

PYTHONPATH=$PYTHONPATH:$PWD python training/main.py \
  experiment=training/experiments_dino_single/opening_fridge_multimodalv2_dinovitgru.py \
  task.name=OpeningfridgeGraspTask \
  task_config=opening_fridge \
  agent=stretch \
  target_object_types=open_fridge \
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
  task_config.knob_reach_distance=0.3 \
  training.small_batch=False \
  task_config.spawn_range_max=3.0 \
  mdp.reward.opening_fridge.train.energy_penalty=-0.01 \
  mdp.reward.opening_fridge.train.open_initial_reward=0.0 \
  mdp.reward.opening_fridge.train.open_section_reward=80.0 \
  output_dir=$OUTPUT_DIR \
  seed=$SEED