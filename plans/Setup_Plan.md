# Plan

## Step 1: Get It Running

**Goal:** Train `k1_amp` on the L40 server with WandB logging.

### Steps

1. **Create virtual environment**
   ```bash
   python3.8 -m venv lr_gym
   source lr_gym/bin/activate
   ```

2. **Install PyTorch**
   ```bash
   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install IsaacGym**
   - Download IsaacGym Preview 4
   - Install it into the parent directory (`../`)

4. **Install this repo**
   ```bash
   pip install -e ".[isaacgym]"
   ```

5. **Log in to WandB**
   ```bash
   wandb login $(cat wandb_key)
   ```

6. **Start training**
   ```bash
   python legged_gym/scripts/train.py --task=k1_amp --headless --sync_wandb
   ```

### Notes
- Tested on L40 server, straightforward setup
- Use `--sync_wandb` to enable WandB run tracking


