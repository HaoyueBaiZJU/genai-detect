import numpy as np
import torch
import torchmetrics
from torchmetrics import AveragePrecision
from utils.metric_utils import get_measures, print_measures

real_ood = torch.load('./ckpt/psnr_list_laion_glide_100_10_lora.pt')
gen_in = torch.load('./ckpt/psnr_list_fake_glide_100_10_lora.pt')

measures = get_measures(gen_in, real_ood, plot=False)
print_measures(measures[0], measures[1], measures[2], 'energy')

predictions = torch.cat([real_ood, gen_in], dim=0)
labels = torch.cat([torch.zeros(len(real_ood)), torch.ones(len(gen_in))], dim=0)
ap_metric = AveragePrecision(num_classes=1, task="binary")

labels = labels.int()
ap_metric.update(predictions, labels)
mAP = ap_metric.compute()

print("mAP", mAP)



