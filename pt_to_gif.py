import visualize
import numpy as np
import torch
frames_t = torch.load("experiments/pred_frames.pt")
frames = frames_t.numpy()
frames_l = [[frames[i, j].transpose(1, 2, 0) for j in range(frames.shape[1])] for i in range(frames.shape[0])]

gt_frames = torch.load("experiments/gtruth_frames.pt").numpy()
gt_frames_l = [[gt_frames[i, j].transpose(1, 2, 0) for j in range(gt_frames.shape[1])] for i in range(gt_frames.shape[0])]
#for i in range(frames.shape[0]):
for i in range(12, 14):
    pred_name = f"visualize_output/pred_{i}.gif"
    gt_name = f"visualize_output/gt_{i}.gif"
    visualize.export_to_gif(frames_l[i], pred_name, 2)
    visualize.export_to_gif(gt_frames_l[i], gt_name, 2)
