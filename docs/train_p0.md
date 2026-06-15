```
lerobot-train \
  --dataset.repo_id=/root/lerobot/Grasp_OrangeToPlate_Dataset \
  --policy.type=pi0 \
  --output_dir=~/output_lerobot_train/pi0_A \
  --job_name=pi0_A \
  --policy.pretrained_path=/root/.cache/modelscope/hub/models/lerobot/pi0_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --steps=10000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8
  ```