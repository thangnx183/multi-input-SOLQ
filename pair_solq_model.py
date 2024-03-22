from PIL import Image
import os
# from car_damage.config.cfg import envs
from util.misc import nested_tensor_from_tensor_list_v2, nested_tensor_from_tensor_list
from datasets import transforms as T
# from models import build_model
from models.fast_solq import build_model
import torch
import numpy as np
import argparse
import cv2
from torchvision import ops
from util import box_ops
from scipy.optimize import linear_sum_assignment

CATEGORIES = [
    'sli_side_turn_light+rf', 'sli_side_turn_light+lf', 'tyre+rf', 'tyre+rb',
    'tyre+lf', 'tyre+lb', 'alloy_wheel+rf', 'alloy_wheel+rb',
    'alloy_wheel+lf', 'alloy_wheel+lb', 'hli_head_light+rf',
    'hli_head_light+lf', 'hood+f', 'fwi_windshield+f', 'flp_front_license_plate+f',
    'door+rf', 'door+rb', 'door+lf', 'door+lb', 'mirror+rf', 'mirror+lf',
    'handle', 'qpa_quarter_panel+rb', 'qpa_quarter_panel+lb', 'fender+rf',
    'fender+lf', 'grille+f', 'fbu_front_bumper+f', 'rocker_panel+r',
    'rocker_panel+l', 'rbu_rear_bumper+b', 'pillar', 'pillar', 'roof',
    'blp_back_license_plate+b', 'window+r', 'window+l', 'rwi_rear_windshield+b',
    'tail_gate+b', 'tli_tail_light+rb', 'tli_tail_light+lb',
    'fbe_fog_light_bezel+rf', 'fbe_fog_light_bezel+lf', 'fli_fog_light+rf',
    'fli_fog_light+lf', 'fuel_tank_door', 'lli_low_bumper_tail_light+lb',
    'lli_low_bumper_tail_light+rb', 'exhaust'
]

normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normalize = T.Compose([
    T.RandomResize([800], max_size=1024),
    normalize,
])


def matching(cost, pad_value):
    h, w = cost.shape
    if h != w:
        cost = cost.numpy()
        cost = np.pad(cost, [(0, (h < w)*(w-h)), (0, (h > w)*(h-w))],
                      'constant', constant_values=pad_value)
        cost = torch.tensor(cost)
    # print(-cost)
    row_ind, col_ind = linear_sum_assignment(-cost)

    return row_ind, col_ind


def ensemble(out1, out2):
    if len(out2['bboxes']) == 0:
        return out1
    boxes1 = torch.as_tensor(out1['bboxes'])
    boxes2 = torch.as_tensor(out2['bboxes'])
    # print(boxes1,boxes2)
    ious, _ = box_ops.box_iou(boxes1, boxes2)
    # print(ious)
    row_ind, col_ind = matching(ious, pad_value=0)
    h, w = ious.shape
    # print(ious)
    # print(row_ind,col_ind)
    for r, c in zip(row_ind, col_ind):
        if r >= h or c >= w:
            continue
        if ious[r, c] > 0.8 and not (out1['labels'][r] == 'grille+f' and out2['labels'][c] == 'fbu_front_bumper+f'):
            # print(out1['labels'][r], out2['labels'][c])
            out1['labels'][r] = out2['labels'][c]

    return out1


def get_args():
    parser = argparse.ArgumentParser(
        'Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names',
                        default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_period', default=10, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--meta_arch', default='solq', type=str)
    parser.add_argument('--num_classes', type=int)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine',
                        default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # VecInst
    parser.add_argument('--with_vector', default=False, action='store_true')
    parser.add_argument('--n_keep', default=256, type=int,
                        help="Number of coeffs to be remained")
    parser.add_argument('--gt_mask_len', default=128, type=int,
                        help="Size of target mask")
    parser.add_argument('--vector_loss_coef', default=0.7, type=float)
    parser.add_argument('--vector_hidden_dim', default=256, type=int,
                        help="Size of the vector embeddings (dimension of the transformer)")
    parser.add_argument('--no_vector_loss_norm',
                        default=False, action='store_true')
    parser.add_argument('--activation', default='relu',
                        type=str, help="Activation function to use")
    parser.add_argument('--checkpoint', default=False, action='store_true')
    parser.add_argument('--vector_start_stage', default=0, type=int)
    parser.add_argument('--num_machines', default=1, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)
    parser.add_argument('--dcn', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--pretrained', default=None,
                        help='resume from checkpoint')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=2,
                        type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=60, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    args = parser.parse_args(['--meta_arch', 'fast_solq', '--with_box_refine', '--two_stage', '--masks',
                             '--num_classes', '50', '--with_vector', '--vector_hidden_dim', '512'])

    return args


class Pair_SOLQ_model():
    def __init__(self, file_name, confident, device='cuda:0'):
        self.confident = confident
        self.device = device
        # file_name = os.path.join(envs.ROOT_DIR, 'models/mutil_input', file_name)

        self.model, self.postprocessors = build_model(get_args())
        checkpoint = torch.load(file_name, map_location='cpu')
        _ = self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()
    
    def inference_filenames(self, image, ref_image, confident=None):
        if confident is not None:
            self.confident = confident
        image = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        w, h = image.size
        size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

        ref_image = cv2.cvtColor(ref_image.copy(),cv2.COLOR_BGR2RGB)
        ref_image = Image.fromarray(ref_image)
        ref_w, ref_h = ref_image.size
        ref_size = torch.stack(
            [torch.as_tensor([int(ref_h), int(ref_w)])], dim=0)

        input_tensor = [normalize(image, None)[0]]
        ref_tensors = [normalize(ref_image, None)[0]]

        input_nest_tensor, ref_nest_tensor = nested_tensor_from_tensor_list_v2(
            input_tensor, ref_tensors)
        input_nest_tensor = input_nest_tensor.to(self.device)
        ref_nest_tensor = ref_nest_tensor.to(self.device)

        outputs, ref_outputs, cache_memory = self.model(
            (input_nest_tensor, ref_nest_tensor), ref_inference=False)

        result = self.postprocessors['bbox'](
            outputs, torch.as_tensor(size).to(self.device))[0]

        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']
        
        idx = ops.nms(boxes, scores, 0.8)
        scores = scores[idx]
        labels = labels[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        idx = scores > self.confident
        scores = scores[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
        masks = masks[idx].detach().cpu().numpy()
        labels = [CATEGORIES[l] for l in labels]

        out = {'scores': scores.tolist(), 'labels': labels, 'bboxes': [b for b in boxes], 'masks': [
            np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}

        # ref_result = self.postprocessors['bbox'](
        #     ref_outputs, torch.as_tensor(ref_size).to(self.device))[0]

        # ref_masks = ref_result['masks']
        # ref_scores = ref_result['scores']
        # ref_labels = ref_result['labels']
        # ref_boxes = ref_result['boxes']

        # idx = ref_scores > self.confident
        # ref_scores = ref_scores[idx].detach().cpu().numpy()
        # ref_labels = ref_labels[idx].detach().cpu().numpy()
        # ref_boxes = ref_boxes[idx].detach().cpu().numpy().astype(np.int32)
        # ref_masks = ref_masks[idx].detach().cpu().numpy()
        # ref_labels = [CATEGORIES[l] for l in ref_labels]

        # ref_out = {'scores': ref_scores.tolist(), 'labels': ref_labels, 'bboxes': [b for b in ref_boxes], 'masks': [
        #     np.squeeze(m, axis=0).astype(np.uint8) for m in ref_masks]}
        return out, cache_memory

    def inference_without_transition(self, image, ref_image, confident=None):
        if confident is not None:
            self.confident = confident
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        w, h = image.size
        size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

        ref_image = cv2.cvtColor(ref_image.copy(), cv2.COLOR_BGR2RGB)
        ref_image = Image.fromarray(ref_image)

        input_tensor = [normalize(image, None)[0]]
        ref_tensors = [normalize(ref_image, None)[0]]

        input_nest_tensor, ref_nest_tensor = nested_tensor_from_tensor_list_v2(
            input_tensor, ref_tensors)
        input_nest_tensor = input_nest_tensor.to(self.device)
        ref_nest_tensor = ref_nest_tensor.to(self.device)

        outputs, ref_memory, ref_memory_add_pos, transition_memory = self.model.forward_without_transition(
            (input_nest_tensor, ref_nest_tensor))

        result = self.postprocessors['bbox'](
            outputs, torch.as_tensor(size).to(self.device))[0]

        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']

        idx = ops.nms(boxes, scores, 0.8)
        scores = scores[idx]
        labels = labels[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        idx = scores > self.confident
        scores = scores[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
        masks = masks[idx].detach().cpu().numpy()
        labels = [CATEGORIES[l] for l in labels]

        out = {'scores': scores.tolist(), 'labels': labels, 'bboxes': [b for b in boxes], 'masks': [
            np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}

        return out, ref_memory, ref_memory_add_pos, transition_memory

    def inference_with_transition(self, image, ref_memory,ref_memory_add_pos, transition_memory, result_transition, confident=None):
        if confident is not None:
            self.confident = confident
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        w, h = image.size
        size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

        input_tensor = [normalize(image, None)[0]]
        # ref_tensors = [normalize(ref_image, None)[0]]

        input_nest_tensor = nested_tensor_from_tensor_list(input_tensor)
        input_nest_tensor = input_nest_tensor.to(self.device)
        # ref_nest_tensor = ref_nest_tensor.to(self.device)

        out_with_transition_memory, out_with_ref_memory = self.model.forward_with_transition_memory(
            [input_nest_tensor], ref_memory,ref_memory_add_pos, transition_memory)

        result_with_ref_memory = self.postprocessors['bbox'](
            out_with_ref_memory, torch.as_tensor(size).to(self.device))[0]

        masks = result_with_ref_memory['masks']
        scores = result_with_ref_memory['scores']
        labels = result_with_ref_memory['labels']
        boxes = result_with_ref_memory['boxes']

        idx = ops.nms(boxes, scores, 0.8)
        scores = scores[idx]
        labels = labels[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        idx = scores > self.confident
        scores = scores[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
        masks = masks[idx].detach().cpu().numpy()
        labels = [CATEGORIES[l] for l in labels]

        out_with_ref_memory_dict = {'scores': scores.tolist(), 'labels': labels, 'bboxes': [b for b in boxes], 'masks': [
            np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}

        result_with_transition_memory = self.postprocessors['bbox'](
            out_with_transition_memory, torch.as_tensor(size).to(self.device))[0]

        masks = result_with_transition_memory['masks']
        scores = result_with_transition_memory['scores']
        labels = result_with_transition_memory['labels']
        boxes = result_with_transition_memory['boxes']

        idx = ops.nms(boxes, scores, 0.8)
        scores = scores[idx]
        labels = labels[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        idx = scores > self.confident
        scores = scores[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
        masks = masks[idx].detach().cpu().numpy()
        labels = [CATEGORIES[l] for l in labels]
        # print('shortcut : ',labels,scores)

        out_with_transition_memory_dict = {'scores': scores.tolist(), 'labels': labels, 'bboxes': [b for b in boxes], 'masks': [
            np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}
        
        filter_id = [i in result_transition['labels'] for i in out_with_transition_memory_dict['labels']]
        out_with_transition_memory_dict['bboxes'] = np.array(out_with_transition_memory_dict['bboxes'])[filter_id].tolist()
        out_with_transition_memory_dict['masks'] = np.array(out_with_transition_memory_dict['masks'])[filter_id].tolist()
        out_with_transition_memory_dict['labels'] = np.array(out_with_transition_memory_dict['labels'])[filter_id].tolist()
        out_with_transition_memory_dict['scores'] = np.array(out_with_transition_memory_dict['scores'])[filter_id].tolist()

        # print('shortcut : ', out_with_transition_memory_dict['labels'],out_with_transition_memory_dict['scores'])
        # print('shortcut : ', out_with_ref_memory_dict['labels'],out_with_ref_memory_dict['scores'])
        out_with_ref_memory_dict = ensemble(
            out_with_ref_memory_dict, out_with_transition_memory_dict)

        return out_with_ref_memory_dict
        
        # return out_with_transition_memory_dict

    def inference_filenames_old(self, image, ref_image, confident=None):
        if confident is not None:
            self.confident = confident
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        w, h = image.size
        size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

        ref_image = cv2.cvtColor(ref_image.copy(), cv2.COLOR_BGR2RGB)
        ref_image = Image.fromarray(ref_image)
        ref_w, ref_h = ref_image.size
        ref_size = torch.stack(
            [torch.as_tensor([int(ref_h), int(ref_w)])], dim=0)

        input_tensor = [normalize(image, None)[0]]
        ref_tensors = [normalize(ref_image, None)[0]]

        input_nest_tensor, ref_nest_tensor = nested_tensor_from_tensor_list_v2(
            input_tensor, ref_tensors)
        input_nest_tensor = input_nest_tensor.to(self.device)
        ref_nest_tensor = ref_nest_tensor.to(self.device)

        outputs, ref_outputs, cache_memory = self.model(
            (input_nest_tensor, ref_nest_tensor), ref_inference=False)

        result = self.postprocessors['bbox'](
            outputs, torch.as_tensor(size).to(self.device))[0]

        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']

        # idx = ops.nms(boxes,scores,0.8)
        # scores = scores[idx]
        # labels = labels[idx]
        # boxes = boxes[idx]
        # masks = masks[idx]

        idx = scores > self.confident
        scores = scores[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
        masks = masks[idx].detach().cpu().numpy()
        labels = [CATEGORIES[l] for l in labels]

        out = {'scores': scores.tolist(), 'labels': labels, 'bboxes': [b for b in boxes], 'masks': [
            np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}

        # ref_result = self.postprocessors['bbox'](
        #     ref_outputs, torch.as_tensor(ref_size).to(self.device))[0]

        # ref_masks = ref_result['masks']
        # ref_scores = ref_result['scores']
        # ref_labels = ref_result['labels']
        # ref_boxes = ref_result['boxes']

        # idx = ref_scores > self.confident
        # ref_scores = ref_scores[idx].detach().cpu().numpy()
        # ref_labels = ref_labels[idx].detach().cpu().numpy()
        # ref_boxes = ref_boxes[idx].detach().cpu().numpy().astype(np.int32)
        # ref_masks = ref_masks[idx].detach().cpu().numpy()
        # ref_labels = [CATEGORIES[l] for l in ref_labels]

        # ref_out = {'scores': ref_scores.tolist(), 'labels': ref_labels, 'bboxes': [b for b in ref_boxes], 'masks': [
        #     np.squeeze(m, axis=0).astype(np.uint8) for m in ref_masks]}
        # return out, ref_out
        return out, cache_memory

    def inference_filename_with_cache(self, image, cache_memory, confident=None):
        # image = Image.open(img_path).convert('RGB')
        if confident is not None:
            self.confident = confident

        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        w, h = image.size
        size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

        input_tensor = [normalize(image, None)[0]]

        input_nest_tensor = nested_tensor_from_tensor_list(input_tensor)
        input_nest_tensor = input_nest_tensor.to(self.device)

        outputs, _, cache_memory = self.model(
            [input_nest_tensor], ref_inference=False, cache_memory=cache_memory)

        result = self.postprocessors['bbox'](
            outputs, torch.as_tensor(size).to(self.device))[0]

        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']

        idx = ops.nms(boxes, scores, 0.8)
        scores = scores[idx]
        labels = labels[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        idx = scores > self.confident

        # print(idx,masks.device,idx)
        masks = masks.to(idx.device)
        # print('scores ',scores.shape,boxes.shape)
        scores = scores[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
        masks = masks[idx].detach().cpu().numpy()
        labels = [CATEGORIES[l] for l in labels]
#
        out = {'scores': scores.tolist(), 'labels': labels, 'bboxes': [b for b in boxes], 'masks': [
            np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}

        return out, cache_memory

    def inference_filename(self, image):
        # image = Image.open(img_path).convert('RGB')
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        w, h = image.size
        size = torch.stack([torch.as_tensor([int(h), int(w)])], dim=0)

        input_tensor = [normalize(image, None)[0]]

        input_nest_tensor = nested_tensor_from_tensor_list(input_tensor)
        input_nest_tensor = input_nest_tensor.to(self.device)

        outputs, _ = self.model([input_nest_tensor])

        result = self.postprocessors['bbox'](
            outputs, torch.as_tensor(size).to(self.device))[0]

        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']

        idx = scores > self.confident

        # print(idx,masks.device,idx)
        masks = masks.to(idx.device)
        scores = scores[idx].detach().cpu().numpy()
        labels = labels[idx].detach().cpu().numpy()
        boxes = boxes[idx].detach().cpu().numpy().astype(np.int32)
        masks = masks[idx].detach().cpu().numpy()
        # labels = [CATEGORIES[l] for l in labels]
#
        out = {'scores': scores.tolist(), 'labels': labels, 'bboxes': [b for b in boxes], 'masks': [
            np.squeeze(m, axis=0).astype(np.uint8) for m in masks]}

        return out
