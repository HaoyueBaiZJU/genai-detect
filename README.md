# Where’s the liability in the Generative Era? Recovery-based Black-Box Detection of AI-Generated Content

This repository contains the source code for [Where’s the liability in the Generative Era? Recovery-based Black-Box Detection of AI-Generated Content](https://arxiv.org/abs/xxx).

## Overview

The recent proliferation of photorealistic images created by generative models has sparked both excitement and concern, as these images are increasingly indistinguishable from real ones to the human eye. While offering new creative and commercial possibilities, the potential for misuse, such as in misinformation and fraud, highlights the need for effective detection methods. Current detection approaches often rely on access to model weights or require extensive collections of real image datasets, limiting their scalability and practical application in real-world scenarios. In this work, we introduce a novel black-box detection framework that requires only API access, sidestepping the need for model weights or large auxiliary datasets.

## Setups

1. Clone this repository 
```bash
git clone https://github.com/HaoyueBaiZJU/genai-detect
cd genai-detect
```

2. Install the necessary libraries

To set up the required environment, install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
pip install git+https://github.com/cloneofsimo/lora.git
```

## Quick Start


### Data Preparation: 

- Dataset for the diffusion models (e.g., LDM/Glide) can be found [here](https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view?usp=drive_link).

- Download and unzip the file into the following directory.

`./real_folder`

`./fake_folder`

`./masks_folder`

### Running the Evaluation

To evaluate a on the dataset, run the following command:

```
python eval_real_vs_fake.py \
    --real_folder /path/to/real_images \
    --fake_folder /path/to/fake_images \
    --seeds 0 \
    --save_location ./ckpt \
    --checkpoint_folder ./my_sd_inpainting_ckpt \
    --num_inference_steps 50 \
    --guidance_scale 7
```

You can test the benchmark results using the script:

`python evaluate_scores.py`

To conduct fine-tuning via lora, run the following command:

`bash inpainting_lora.sh`


## Citation

If you use our codebase, please cite our work:
