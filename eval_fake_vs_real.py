import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from typing import Tuple
from sklearn.metrics import roc_curve, auc

from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, AveragePrecision
from torchvision.transforms.functional import pil_to_tensor

from lora_diffusion import patch_pipe, tune_lora_scale
from diffusers import StableDiffusionInpaintPipeline

from utils.metric_utils import get_measures, print_measures


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def evaluate_single_image(
    image_path: str,
    pipe,
    prompt: str,
    mask_image=None,
    seeds=0,
    num_gen=3,
    device="cuda",
    num_inference_steps=50,
    guidance_scale=7
):
    """
    Runs inference multiple times on a single image, averages the outputs,
    and computes metrics (SSIM, PSNR, L1, L2) compared to the original image.
    
    Returns a dict of these metrics.
    """
    
    # Load image
    image = Image.open(image_path).convert("RGB").resize((512, 512))

    gt_tensor = pil_to_tensor(image).float().to(device) 

    # Optionally generate it 3 times, then average.
    gt_sample_stack = torch.stack([gt_tensor, gt_tensor, gt_tensor], dim=0)
    mean_gt_sample = gt_sample_stack.mean(dim=0)  # shape [C, H, W]

    output_tensors = []
    for i in range(num_gen):
        torch.manual_seed(seeds + i)
        output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        output_image = output.images[0]

        output_tensor = pil_to_tensor(output_image).float().to(device) 
        output_tensors.append(output_tensor)

    stacked_outputs = torch.stack(output_tensors, dim=0)
    mean_outputs = stacked_outputs.mean(dim=0)

    # Compute metrics
    ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    ms_ssim_val = ms_ssim_fn(
        mean_gt_sample.unsqueeze(0), mean_outputs.unsqueeze(0)
    )  

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr_val = psnr_fn(
        mean_gt_sample.unsqueeze(0), mean_outputs.unsqueeze(0)
    )  

    l1_loss = nn.L1Loss()(mean_gt_sample, mean_outputs)
    l2_loss = nn.MSELoss()(mean_gt_sample, mean_outputs)

    return {
        "ms_ssim": ms_ssim_val.item(),
        "psnr":    psnr_val.item(),
        "l1":      l1_loss.item(),
        "l2":      l2_loss.item()
    }


def evaluate_folder(
    folder_path: str,
    pipe,
    prompt: str,
    device="cuda",
    seeds=0,
    num_gen=3,
    mask_image=None,
    num_inference_steps=50,
    guidance_scale=7,
    save_tensors_prefix=None
):
    """
    Evaluates a folder of images:
    - For each image, runs evaluate_single_image
    - Aggregates MS-SSIM, PSNR, L1, and L2 over all images
    - Optionally saves results in .pt files if save_tensors_prefix is given
    """
    ssim_list = []
    psnr_list = []
    l1_list   = []
    l2_list   = []

    valid_exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
    filenames = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)
    ]
    filenames.sort()

    print(f"[INFO] Evaluating folder: {folder_path}  ({len(filenames)} images)")
    for idx, filename in enumerate(filenames):
        image_path = os.path.join(folder_path, filename)
        print(f"  -> [{idx+1}/{len(filenames)}] {filename}")

        metrics = evaluate_single_image(
            image_path=image_path,
            pipe=pipe,
            prompt=prompt,
            mask_image=mask_image,
            seeds=seeds,
            num_gen=num_gen
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

        ssim_list.append(metrics["ms_ssim"])
        psnr_list.append(metrics["psnr"])
        l1_list.append(metrics["l1"])
        l2_list.append(metrics["l2"])

    # Convert to numpy
    ssim_arr = np.array(ssim_list)
    psnr_arr = np.array(psnr_list)
    l1_arr   = np.array(l1_list)
    l2_arr   = np.array(l2_list)

    # Print out average values
    print(f"[RESULT] Folder: {folder_path}")
    print("  MS-SSIM (avg):", ssim_arr.mean())
    print("  PSNR    (avg):", psnr_arr.mean())
    print("  L1      (avg):", l1_arr.mean())
    print("  L2      (avg):", l2_arr.mean())

    # Optionally save
    if save_tensors_prefix is not None:
        torch.save(torch.from_numpy(ssim_arr), f"{save_tensors_prefix}_ssim.pt")
        torch.save(torch.from_numpy(psnr_arr), f"{save_tensors_prefix}_psnr.pt")
        torch.save(torch.from_numpy(l1_arr),   f"{save_tensors_prefix}_l1.pt")
        torch.save(torch.from_numpy(l2_arr),   f"{save_tensors_prefix}_l2.pt")

    return ssim_arr, psnr_arr, l1_arr, l2_arr


# Main function with arg parsing

def main(args):
    """
    1. Loading pipeline
    2. Evaluating real & fake
    3. Computing scores and metrcs.
    """

    torch.manual_seed(args.seeds)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
    ).to(args.device)

    # safety_checker:
    def dummy_checker(images, **kwargs):
        return images, False
    # pipe.safety_checker = dummy_checker

    # Evaluate real images
    real_ssim, real_psnr, real_l1, real_l2 = evaluate_folder(
        folder_path=args.real_folder,
        pipe=pipe,  
        prompt=args.prompt,
        device=args.device,
        seeds=args.seeds,
        num_gen=args.num_gen,
        mask_image=None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        save_tensors_prefix=os.path.join(args.save_location, "real_metrics") if args.save_location else None
    )

    # Evaluate fake images
    fake_ssim, fake_psnr, fake_l1, fake_l2 = evaluate_folder(
        folder_path=args.fake_folder,
        pipe=pipe,  # pipe if you have it loaded
        prompt=args.prompt,
        device=args.device,
        seeds=args.seeds,
        num_gen=args.num_gen,
        mask_image=None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        save_tensors_prefix=os.path.join(args.save_location, "fake_metrics") if args.save_location else None
    )

    # Evaluate Example: Compute AP & AUROC based on PSNR

    measures = get_measures(fake_psnr, real_psnr, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'energy')

    # Merics presented in the paper
    predictions = torch.cat([real_psnr, fake_psnr], dim=0)
    labels = torch.cat([torch.zeros(len(real_psnr)), torch.ones(len(fake_psnr))], dim=0).int()
    ap_metric = AveragePrecision(num_classes=1, task="binary")

    ap_metric.update(predictions, labels)
    mAP = ap_metric.compute()
    print("mAP", mAP)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for Real vs Fake Images")

    # Device selection
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g. 'cpu', 'cuda').")
    parser.add_argument("--lora_scale", type=float, default=0.3, help="Lora_scale=0 refers to not using fine-tuning.")

    # Folder paths
    parser.add_argument("--real_folder", type=str, required=True, help="Folder containing real images.")
    parser.add_argument("--fake_folder", type=str, required=True, help="Folder containing fake images.")
    parser.add_argument("--mask_image_path", type=str, required="", help="Mask image path.")

    parser.add_argument("--save_location", type=str, default=None, help="Folder to save metric tensors (e.g. 'ckpt/'). If None, does not save.")
    parser.add_argument("--prompt",ctype=str, default="photo of <s1><s2>", help="Prompt string for the inpainting or generation pipeline.")
    parser.add_argument("--seeds", type=str, default="0", help="Random seeds.")
    parser.add_argument("--num_gen", type=int, default=3, help="Number of generations.")

    parser.add_argument("--base_model", type=str, default="botp/stable-diffusion-v1-5-inpainting", help="Path to the base model checkpoint.")
    parser.add_argument("--lora_model", type=str, default="./exps/step_3000.safetensors", help="Path to the lora model checkpoint.")

    # Inference parameters
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps for the pipeline.")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale for the pipeline.")

    args = parser.parse_args()
    main(args)

