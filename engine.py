# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import json
import numpy as np

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher, single_data_prefetcher
import datasets.transforms as T
from util.misc import nested_tensor_from_tensor_list_v2
import cv2
import time

normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normalize = T.Compose([
    T.RandomResize([800], max_size=1024),
    normalize,
])


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,input_mode='multi'):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if input_mode == 'multi':
        prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    else:
        prefetcher = single_data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(
                model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox')
                      if k in postprocessors.keys())
    iou_types = tuple(k for k in ('bbox',)
                      if k in postprocessors.keys())
    print('tupe iou ',iou_types)
    # iou_types = tuple('bbox',)
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator = None
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = list(samples)
        samples[0] = samples[0].to(device)
        if len(samples) == 2:
            samples[1] = samples[1].to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        t1 = time.time()
        outputs = model(samples)
        t2 = time.time()
        loss_dict = criterion(outputs, targets)
        t3 = time.time()
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        t4 = time.time()

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0)
        
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        t5 = time.time()
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](
                results, outputs, orig_target_sizes, target_sizes)
        t6 = time.time()
        res = {target['image_id'].item(): output for target,
               output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        t7 = time.time()

        print('execute : ',t2-t1, ' loss cal : ',t3-t2,'log : ',t4-t3, 'post box',t5-t4,' post mask ',t6-t5,'coco update ',t7-t6)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        # if 'segm' in postprocessors.keys():
        #     stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def single_inference(model, postprocessors, image, ref_image, device, thres, cates=None):
    model.eval()
    # criterion.eval()
    w, h = image.size
    mask = np.zeros((h, w)).astype(np.int32)
    # ref_w,ref_h = ref_image.size
    size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

    input_tensor = [normalize(image, None)[0]]
    ref_tensors = [normalize(ref_image, None)[0]]

    input_nest_tensor, ref_nest_tensor = nested_tensor_from_tensor_list_v2(
        input_tensor, ref_tensors)
    input_nest_tensor = input_nest_tensor.to(device)
    ref_nest_tensor = ref_nest_tensor.to(device)

    # print('debug input : ',input_tensor[0].sum(),ref_tensors[0].sum())
    #  tensor(4394974.50000) tensor(1626054.37500)

    # with torch.no_grad():
    import time

    t1 = time.time()
    outputs = model((input_nest_tensor, ref_nest_tensor))
    t2 = time.time()

    print('time infer alone ,', t2-t1)
    # print('logit :',outputs['pred_logits'].sum())

    result = postprocessors['bbox'](
        outputs, torch.as_tensor(size).to(device))[0]
    # print(result)
    if 'segm' in postprocessors.keys():
        # target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        result = postprocessors['segm']([result], outputs, size, size)[0]
        masks = result['masks']

    scores = result['scores']
    labels = result['labels']
    boxes = result['boxes']

    # print(labels,scores, thres)

    idx = scores > thres
    scores = scores[idx].cpu().detach().numpy()
    labels = labels[idx].cpu().detach().numpy()
    boxes = boxes[idx].cpu().detach().numpy().astype(np.int32)

    # print(scores)

    if 'segm' in postprocessors.keys():
        masks = masks[idx].cpu().detach().numpy()
    else:
        masks = []
        for b in boxes:
            # print(b)
            b_mask = mask.copy()
            b_mask[b[1]:b[3], b[0]:b[2]] = 1
            masks.append(b_mask)

    image = np.array(image)
    # image = cv2.copyMakeBorder(image, 0, ref_h-h, 0, ref_w - w, cv2.BORDER_CONSTANT,value=(127,127,127))
    # ref_image = np.array(ref_image)

    # image =np.hstack((image,ref_image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for score, label, box in zip(scores, labels, boxes):
        # print(box[:2].astype(np.int32),box[2:],box)
        # print(cates[label],str(round(score,2)))
        box = box.astype(np.int32)
        image = cv2.rectangle(image, tuple(
            box[:2]), tuple(box[2:]), (255, 0, 0), 1)
        if cates is not None:
            image = cv2.putText(image, str(round(score, 2)) + '|' + cates[label], (
                box[0], (box[3]+box[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    if 'segm' in postprocessors.keys():
        for mask in masks:
            # print(mask.shape)
            mask = np.squeeze(mask, axis=0).astype(np.uint8)
            cons, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, cons, -1, (255, 0, 0), 2)
            # break
    # cv2.imwrite('demo.png',image)
    # cv2.imwrite('demo.png',image)

    # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
    return {'scores': scores, 'labels': labels, 'boxes': boxes, 'masks': masks}, image

# @torch.no_grad()


def solq_single_inference(model, postprocessors, image, ref_image, device, thres, cates=None):
    w, h = image.size
    size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

    input_tensor = [normalize(image, None)[0]]
    ref_tensors = [normalize(ref_image, None)[0]]

    input_nest_tensor, ref_nest_tensor = nested_tensor_from_tensor_list_v2(
        input_tensor, ref_tensors)
    input_nest_tensor = input_nest_tensor.to(device)
    ref_nest_tensor = ref_nest_tensor.to(device)

    # t1 = time.time()
    outputs, ref_outputs = model((input_nest_tensor, ref_nest_tensor),ref_inference=False)
    # outputs, ref_outputs = model([input_nest_tensor],ref_inference=False)



    # t2 = time.time()
    # print('time infer alone ,', t2-t1)

    result = postprocessors['bbox'](
        outputs, torch.as_tensor(size).to(device))[0]

    # t3 = time.time()
    # print('time post process ,', t3-t2)

    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    boxes = result['boxes']

    idx = scores > thres
    masks = masks.to(idx.device)
    print(idx.device)
    print(masks.device)
    scores = scores[idx].detach().cpu().numpy()
    labels = labels[idx].detach().cpu().numpy()
    boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
    masks = masks[idx].detach().cpu().numpy()

    if ref_outputs is not None:
        ref_w, ref_h = ref_image.size
        ref_size = torch.stack([torch.as_tensor([int(ref_h), int(ref_w)])], dim=0)
        ref_result = postprocessors['bbox'](ref_outputs,torch.as_tensor(ref_size).to(device))[0]
        
        ref_masks = ref_result['masks']
        ref_scores = ref_result['scores']
        ref_labels = ref_result['labels']
        ref_boxes = ref_result['boxes']
        
        idx = ref_scores > thres
        ref_scores = ref_scores[idx].detach().cpu().numpy()
        ref_labels = ref_labels[idx].detach().cpu().numpy()
        ref_boxes = ref_boxes[idx].detach().cpu().numpy().astype(np.int32)
        ref_masks = ref_masks[idx].detach().cpu().numpy()
        
        ref_out = {'scores': ref_scores, 'labels': ref_labels, 'boxes': ref_boxes, 'masks': [np.squeeze(m, axis=0).astype(np.uint8) for m in ref_masks]}
    else:
        ref_out = None

    return {'scores': scores, 'labels': labels, 'boxes': boxes, 'masks': [np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}, ref_out