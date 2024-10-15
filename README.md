# Harmonic Mobile Manipulation

This is repo for our work:

Harmonic Mobile Manipulation, IROS 2024 (Oral)

Ruihan Yang<sup>1</sup>, Yejin Kim<sup>2</sup>, Rose Hendrix<sup>2</sup>, Aniruddha Kembhavi<sup>2,3</sup>, Xiaolong Wang<sup>1</sup>, Kiana Ehsani<sup>2</sup>

<sup>1</sup>UC San Diego / <sup>2</sup>PRIOR @ Allen Institute for AI / <sup>3</sup>University of Washington, Seattle

[Project Page](https://rchalyang.github.io/HarmonicMM) / [Arxiv](https://arxiv.org/abs/2312.06639)

![img](./figures/teaser_v9.svg)

In this work, we address diverse mobile manipulation tasks integral to human's daily life. Trained in a photo-realistic simulation,  Our controller effectively accomplishes tasks through harmonious mobile manipulation in a real-world apartment featuring a novel layout, without any fine-tuning or adaptation

## Installation

```bash

# Similar Script is provided in build_env.sh
export MY_ENV_NAME=HarmonicMM
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"

conda create --name $MY_ENV_NAME python=3.8
conda activate $MY_ENV_NAME

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git@3b473b0e682c091a9e53623eebc1ca1657385717
```

You can also build the environment with dockerfile provided in *docker* directory. Prebuilt docker image `rchal97/mobile_manipulation` can be found on dockerhub 

## Run

### Training
To run experiments, use:

```
# The hyperparameters are in the `config/` folder. 

# You might need to change the number of environments according to the number of GPUs used and GPU memory.

# Open Door Pull
bash scripts/training/open_door_pull.sh

# Open Door Push
bash scripts/training/open_door_push.sh

# Clean Table
bash scripts/training/clean_table.sh

# Open Fridge
bash scripts/training/open_fridge.sh


# If running into issue with DINOv2 model, consider run the following command to download DINOv2 model first.

python test_scripts/download_dino_model.py
```

for logging, you need to add wandb entity in `config/main.yaml`

### Evaluation

To only run evaluation (validation / testing), pass in the `eval=true` flag and a path to the `checkpoint=<checkpoint>.pth` PyTorch file.



## Bibtex 

```
@misc{yang2023harmonic,
          title={Harmonic Mobile Manipulation}, 
          author={Ruihan Yang and Yejin Kim and Aniruddha Kembhavi and Xiaolong Wang and Kiana Ehsani},
          year={2023},
          eprint={2312.06639},
          archivePrefix={arXiv},
          primaryClass={cs.RO}
}
```

## Possible Issues:

#### Vulkan:

If vulkan info doesn't provide correct information inside of container
```
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

#### 