python3 training/main.py \
  experiment=training/evaluation/robothor_test.py \
  agent=locobot \
  target_object_types=robothor_habitat2022 \
  wandb.project=procthor-training \
  machine.num_train_processes=40 \
  ai2thor.platform=Linux64 \
  model.add_prev_actions_embedding=true \
  wandb.name=100-layouts \
  procthor.num_train_houses=100
