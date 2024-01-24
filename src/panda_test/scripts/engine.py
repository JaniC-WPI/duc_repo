import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils as utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

# def get_adjacency_matrix(num_keypoints):
#     A = torch.zeros(num_keypoints, num_keypoints)
#     for i in range(num_keypoints-1):
#         A[i,i+1] = 1
#         A[i+1,i] = 1
#     return A

# def get_laplacian(A):
#     D = torch.diag(A.sum(1))
#     return D - A


# def graph_laplacian_loss(batch_keypoints, L):
#     batch_loss = 0
#     for keypoints in batch_keypoints:
#         keypoints = keypoints.cuda().squeeze()
        
#         # Print after squeezing:
# #         print(f"Shape after squeezing: {keypoints.shape}")

#         # If keypoints is still 3D after squeezing, then reshape it
#         if keypoints.dim() == 3:
#             if keypoints.shape[0] == 6 and keypoints.shape[1] == 6:
#                 # Add logic to handle or skip this specific case
# #                 print("Encountered an unexpected tensor shape. Skipping this tensor...")
#                 continue
#             keypoints = keypoints.squeeze(1)
#             # If it's still 3D after the above operation, drop an extra dimension.
#             if keypoints.dim() == 3:
#                 keypoints = keypoints.squeeze(0)

#         # Print before matrix multiplication:
# #         print(f"Shape before matrix multiplication: {keypoints.shape}")

#         keypoints = keypoints[:, :2]  # exclude visibility
        
#         # Perform matrix multiplication
#         batch_loss += torch.mm(torch.mm(keypoints.t(), L), keypoints).trace()

#     return batch_loss

def match_keypoints(pred_keypoints, gt_keypoints):
    # Ensure pred_keypoints is not empty
    if len(pred_keypoints) == 0:
        return gt_keypoints.clone()
    
    # If number of predicted keypoints is less than ground truth keypoints
    while len(pred_keypoints) < len(gt_keypoints):
        pred_keypoints = torch.cat([pred_keypoints, pred_keypoints], dim=0)
    
    # If number of predicted keypoints is more than ground truth keypoints
    pred_keypoints = pred_keypoints[:len(gt_keypoints)]
    
    return pred_keypoints

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
            
        # Extract predicted keypoints for all images in the batch
        with torch.no_grad():
            model.eval()
            outputs = model(images)
            model.train()

        # Extract ground truth keypoints for all images in the batch
        gt_keypoints_list = [target['keypoints'] for target in targets]
        
        # Match the predicted keypoints with ground truth keypoints
        pred_keypoints_list = []
        for i, output in enumerate(outputs):
            matched_keypoints = match_keypoints(output['keypoints'], gt_keypoints_list[i])
            pred_keypoints_list.append(matched_keypoints)
        
        # Compute the graph Laplacian loss for the keypoints
        gl_loss = graph_laplacian_loss(gt_keypoints_list, L) + graph_laplacian_loss(pred_keypoints_list, L)

        # Combine the graph Laplacian loss with the model's loss
        alpha = 0.1  # hyperparameter to weight the importance of the graph Laplacian loss
        total_loss = sum(loss for loss in loss_dict.values()) + alpha * gl_loss

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, lambda_laplacian=1.0):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = f"Epoch: [{epoch}]"

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=warmup_factor, total_iters=warmup_iters
#         )

#     # Prepare the Laplacian matrix. Assuming 6 keypoints (adjust as needed)
#     A = get_adjacency_matrix(6)  # Adjust 6 as per the number of keypoints
#     L = get_laplacian(A).cuda()

#     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         loss_dict = model(images, targets)

#         with torch.no_grad():
#             model.eval()
#             outputs = model(images)
#             model.train()

#         gt_keypoints_list = [target['keypoints'] for target in targets]

#         pred_keypoints_list = []
#         for i, output in enumerate(outputs):
#             matched_keypoints = match_keypoints(output['keypoints'], gt_keypoints_list[i])
#             pred_keypoints_list.append(matched_keypoints)

#         gl_loss = graph_laplacian_loss(gt_keypoints_list, L) + graph_laplacian_loss(pred_keypoints_list, L)

#         alpha = lambda_laplacian  # Adjusted to use the lambda_laplacian parameter
#         total_loss = sum(loss for loss in loss_dict.values()) + alpha * gl_loss
        
#         if not math.isfinite(total_loss.item()):
#             print(f"Loss is {total_loss.item()}, stopping training")
#             sys.exit(1)

#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(graph_loss=gl_loss.item(), lr=optimizer.param_groups[0]["lr"])

#     return metric_logger

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator