# from datasets import build_dataset

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import torch
from engine import solq_single_inference
# from models.deformable_detr import PostProcess
from PIL import Image
import json
from pathlib import Path
import argparse
from models.solq import build_model
import numpy as np
import cv2
from util import box_ops
from scipy.optimize import linear_sum_assignment
import os
# from segment_anything import sam_model_registry, SamPredictor
import time

global TOTAL_BOX
TOTAL_BOX = 0
global MATCHING_BOX
MATCHING_BOX = 0

# CATEGORIES = ['sli_side_turn_light', 'tyre', 'alloy_wheel', 'hli_head_light', 'hood', 'fwi_windshield', 'flp_front_license_plate', 'door', 'mirror', 'handle', 'qpa_quarter_panel', 'fender', 'grille', 'fbu_front_bumper', 'rocker_panel', 'rbu_rear_bumper', 'roof', 'blp_back_license_plate', 'window', 'rwi_rear_windshield', 'tail_gate', 'tli_tail_light', 'fbe_fog_light_bezel', 'fli_fog_light', 'fuel_tank_door', 'lli_low_bumper_tail_light']
CATEGORIES = [
    'sli+rf', 'sli+lf', 'tyre+rf', 'tyre+rb',
    'tyre+lf', 'tyre+lb', 'alloy_wheel+rf', 'alloy_wheel+rb',
    'alloy_wheel+lf', 'alloy_wheel+lb', 'hli+rf',
    'hli+lf', 'hood+f', 'fwi+f', 'flp+f',
    'door+rf', 'door+rb', 'door+lf', 'door+lb', 'mirror+rf', 'mirror+lf',
    'handle', 'qpa+rb', 'qpa+lb', 'fender+rf',
    'fender+lf', 'grille+f', 'fbu+f', 'rocker_panel+r',
    'rocker_panel+l', 'rbu+b', 'pillar', 'pillar', 'roof',
    'blp+b', 'window+r', 'window+l', 'rwi+b',
    'tail_gate+b', 'tli+rb', 'tli+lb',
    'fbe+rf', 'fbe+lf', 'fli+rf',
    'fli+lf', 'fuel_tank_door', 'lli+lb',
    'lli+rb', 'exhaust'
]
print(len(CATEGORIES))
CTI = dict([(c, idx) for idx, c in enumerate(CATEGORIES)])

np.random.seed(42)
mask_colors = [
    np.random.randint(0, 256, (1, 3), dtype=np.uint8)[0]
    for _ in range(len(CATEGORIES))
]

data = json.load(open('coco_data/carpart-side/crop/test.json'))
data_path = Path('coco_data/carpart-side/crop')
ref_data_path = Path('coco_data/carpart-side/ref')

detr_path = 'checkpoints/video-side-carpart-side-ft.pth'
# maskrcnn_path = ['checkpoints/maskrcnn-cp/carpart_rear.py','checkpoints/maskrcnn-cp/epoch_29.pth']
maskrcnn_path = ['checkpoints/carpart_20221030_configs.py',
                 'checkpoints/carpart_20221030_model.pth']
# maskrcnn_path = ['checkpoints/maskrcnn-cp/carpart_20230302_configs.py','checkpoints/maskrcnn-cp/carpart_20230302_model.pth']


def init_pair_model(pair_input_path, device):
    def get_args():
        parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
        parser.add_argument('--lr', default=2e-4, type=float)
        parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
        parser.add_argument('--lr_backbone', default=2e-5, type=float)
        parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
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
        parser.add_argument('--num_classes',type=int)


        parser.add_argument('--sgd', action='store_true')

        # Variants of Deformable DETR
        parser.add_argument('--with_box_refine', default=False, action='store_true')
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
        parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
        parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
        parser.add_argument('--checkpoint', default=False, action='store_true')
        parser.add_argument('--vector_start_stage', default=0, type=int)
        parser.add_argument('--num_machines', default=1, type=int)
        parser.add_argument('--loss_type', default='l1', type=str)
        parser.add_argument('--dcn', default=False, action='store_true')

        # Model parameters
        parser.add_argument('--frozen_weights', type=str, default=None,
                            help="Path to the pretrained model. If set, only the mask head will be trained")
        parser.add_argument('--pretrained', default=None, help='resume from checkpoint')

        # * Backbone
        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                            help="position / size * scale")
        parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')

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

        args = parser.parse_args(['--with_box_refine','--two_stage','--masks','--num_classes','50','--with_vector','--vector_hidden_dim','256'])
        
        return args

    pair_model, postprocess = build_model(get_args())
    
    checkpoint = torch.load(pair_input_path, map_location='cpu')
    out = pair_model.load_state_dict(checkpoint['model'], strict=True)
    print('loading ',out)
    pair_model = pair_model.to(device)
    pair_model.eval()
    

    print('pair model loaded')

    return pair_model, postprocess


def pair_model_inference(pair_model, postprocess, images, device, thres=0.1, cates=None):
    return solq_single_inference(pair_model, postprocess, images[0], images[1], device, thres, cates)


def mask_model_inference(mask_model, image, device, thres, cates=None):
    np_image = np.array(image)
    result = inference_detector(mask_model, np_image)
    img_, pred_boxes, pred_segms, pred_labels, scores = show_result_pyplot(
        mask_model, np_image.copy(), result, score_thr=0.3)
    pred_boxes = np.array(pred_boxes).reshape(-1, 4)

    if len(pred_boxes) == 0:
        pred_segms = np.array([])

    pred_labels = [mask_model.CLASSES[l] for l in pred_labels]

    return {'boxes': pred_boxes, 'labels': pred_labels, 'scores': scores, 'masks': pred_segms}, img_


def vis(image, result):
    draw_image = image.copy()
    # print('1')
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)

    for b, l, c, m in zip(result['boxes'], result['labels'], result['scores'], result['masks']):
        clr = [255,0,0]
        
        if CATEGORIES[l] == 'hli+lf':
            clr = [255,255,0]
        elif CATEGORIES[l] == 'fbu+f':
            clr = [2550,0,255]
        
        # print(CATEGORIES[l],c)
        b = list(map(int, b))
        cv2.rectangle(draw_image, tuple(b[:2]), tuple(
            b[2:]), (102, 220, 225), thickness=2)

        if isinstance(l, str):
            cv2.putText(draw_image, l+'|'+str(round(c, 2)),
                        (b[0]//2+b[2]//2, b[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1)
        else:
            cv2.putText(draw_image, CATEGORIES[l]+'|'+str(round(
                c, 2)), (b[0]//2+b[2]//2, b[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1)

        # print(m.shape)
        m = np.array(m).astype(np.uint8)
        _, cons, _ = cv2.findContours(
            m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw_image = cv2.drawContours(draw_image, cons, -1, clr, 2)

    return draw_image


def matching(cost, pad_value):
    h, w = cost.shape
    if h != w:
        cost = cost.numpy()
        cost = np.pad(cost, [(0, (h < w)*(w-h)), (0, (h > w)*(h-w))],
                      'constant', constant_values=pad_value)
        cost = torch.tensor(cost)

    row_ind, col_ind = linear_sum_assignment(-cost)

    return row_ind, col_ind


def ensemble(pair_model, postprocess, sam_predictor, mask_model, images, thres, device):
    # t1 = time.time()
    pair_result = pair_model_inference(
        pair_model, postprocess, images, device, thres)
    
    draw  = vis(np.array(images[0]),pair_result[0])
    ref_draw = vis(np.array(images[1]),pair_result[1])

    # t2  = time.time()
    # print(' detr : ',t2-t1)

    # cv2.imwrite('demo_pair.jpg',draw1)

    return pair_result[0],[draw,ref_draw],False

    # print('pair model :',[CATEGORIES[l] for l in pair_result['labels']])
    # mask_result, draw_ = mask_model_inference(
    #     mask_model, images[0], device, 0.1)

    # t3 = time.time()
    
    # print('mask rcnn : ',t3-t2,'detr : ',t2-t1)
    # # draw_final = vis(np.array(images[0]), pair_result)
    # draw_final = np.array(images[0])

    # # cv2.imwrite('demo_final.jpg',draw_final)
    # # draw_final = np.hstack((draw,draw_final))

    # return pair_result, draw_final, False


def segment(sam_predictor, image, result, device):

    # print( result['boxes'])
    # box = result['boxes'][0]
    # box = np.array(box)
    if len(result['labels']) == 0:
        return image

    sam_predictor.set_image(image)
    boxes = torch.as_tensor(result['boxes'], device=device)
    boxes = sam_predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    # label = CATEGORIES[result['labels'][0]]
    center_points = []
    # b_point_labels = []
    for idx, box in enumerate(result['boxes']):
        point = (box[:2] + box[2:])/2
        print(box, point)
        center_points.append(point)

    b_points = []
    b_labels = []
    center_points = np.array(center_points).reshape(-1, 2)
    print('center poinst ', center_points)

    for i in range(len(center_points)):
        points = np.array(center_points.copy())
        labels = [0 for j in center_points]
        labels[i] = 1

        b_points.append(points)
        b_labels.append(labels)
    # print(b_points[0].shape)
    b_points = np.array(b_points).reshape(-1, len(center_points), 2)
    b_points = torch.as_tensor(b_points, device=device)
    b_labels = np.array(b_labels).reshape(-1, len(center_points))
    b_labels = torch.as_tensor(b_labels, device=device)

    masks, score, _ = sam_predictor.predict_torch(
        point_coords=b_points,
        point_labels=b_labels,
        boxes=boxes,
        multimask_output=True,
    )

    print(masks.shape, score)
    # id_max = np.argmax(score)
    # print(score,id_max)
    for idx, mask in enumerate(masks):
        if idx != 6:
            continue
        print(CATEGORIES[result['labels'][idx]])
        mask = mask.sum(dim=0) > 0
        # print(mask.shape)

        mask = mask.cpu().numpy()
        # image[mask] = image[mask]*0.5+0.5*np.array([255,0,255])

        mask = np.array(mask).astype(np.uint8)
        _, cons, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        con = max(cons, key=cv2.contourArea)
        cv2.drawContours(image, [con], -1, (0, 100, 255), 2)

    cv2.imwrite('demo_sam_2.jpg', image)

    # print(box,label)
    return image


def evaluate_result(gt_boxes, gt_labels, result):
    # print('pred  : ',[CATEGORIES[i] for i in result['labels']])
    # print('gt : ',[CATEGORIES[i] for i in gt_labels])

    pred_boxes = torch.as_tensor(result['boxes'])

    gt_boxes = box_ops.box_xywh_to_xyxy(torch.as_tensor(gt_boxes))

    if len(pred_boxes) == 0:
        return [], [], [i for i in range(len(gt_labels))]
    ious, _ = box_ops.box_iou(pred_boxes, gt_boxes)

    gt_labels = torch.as_tensor(gt_labels)
    match_label = torch.stack(
        [gt_labels == i for i in result['labels']]).type(torch.LongTensor)
    cost = ious+match_label
    h, w = cost.shape
    row_ind, col_ind = matching(cost, pad_value=1)
    # print(ious,row_ind,col_ind)

    tp, fp, fn = [], [], []

    for r, c in zip(row_ind, col_ind):
        # FP
        if c >= w:
            fp.append(r)
            # print('fp : ',CATEGORIES[result['labels'][r]])
            continue

        # FN
        if r >= h:
            fn.append(c)
            # print('fn : ',CATEGORIES[gt_labels[c]])
            continue

        if ious[r, c] > 0.2 and match_label[r, c] == 1:  # TP
            tp.append(r)
        else:  # FP
            fp.append(r)
            fn.append(c)
            # print('fp : ',CATEGORIES[result['labels'][r]],CATEGORIES[gt_labels[c]])

    return tp, fp, fn


def main():
    device = torch.device('cuda:0')
    pair_model, postprocess = init_pair_model(detr_path, device)

    # mask_model = init_detector(maskrcnn_path[0], maskrcnn_path[1], device)
    mask_model = None
    # sam_checkpoint = "sam_vit_b_01ec64.pth"
    # model_type = "vit_b"
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    # sam_predictor = SamPredictor(sam)
    sam_predictor = None

    out_ens_path = 'output_ensemble_side'
    os.system('rm -rf '+out_ens_path)

    # return

    Path(out_ens_path).mkdir(parents=True, exist_ok=True)
    (Path(out_ens_path)/'tp').mkdir(parents=True, exist_ok=True)
    (Path(out_ens_path)/'fp').mkdir(parents=True, exist_ok=True)
    (Path(out_ens_path)/'fn').mkdir(parents=True, exist_ok=True)

    (Path(out_ens_path)/'check'/'tp').mkdir(parents=True, exist_ok=True)
    (Path(out_ens_path)/'check'/'fp').mkdir(parents=True, exist_ok=True)

    total_tp, total_fp, total_fn = 0, 0, 0

    for idx,img in enumerate(data['images']):
        
        # case_id = img['case_id']
        file_name = img['file_name']
        # print(file_name) output_ensemble_side/fp/https:__generalide.motionscloud.com_rails_active_storage_blobs_eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBamppIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--9f5a53eda0f474f391bb65b5a1315c553f64d294_8CADBF91-351D-430F-B675-860813B5753E12_crop.jpeg
        # output_ensemble_side/fp/https:__s3.amazonaws.com_mc-ai_dataset_india_20190409_imgs_1255_DSC081291_crop.JPG
        # output_ensemble_side/fp/https:__s3.amazonaws.com_mc-ai_dataset_india_20190409_imgs_1043_DSCN387312_crop.JPG
        if file_name != 'https:__generalide.motionscloud.com_rails_active_storage_blobs_eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBaGVsIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--72b71c72d7b2071f036f8e7a94160e0f434b7484_20210609_13355212_crop.jpg':
            continue
        ori_file_name = file_name.replace('_crop', '')
        image = Image.open(data_path/file_name).convert('RGB')
        ref_image = Image.open(ref_data_path/ori_file_name).convert('RGB')

        image = Image.open('input/zoomin_time_1688356915411_reftime_1688356910826.jpg').convert('RGB')
        ref_image = Image.open('input/ref_time_1688356910826.jpg').convert('RGB')

        result, draw, check = ensemble(pair_model, postprocess, sam_predictor, mask_model, [
                                       image, ref_image], thres=0.4, device=device)

        cv2.imwrite('demo_mask.jpg', draw[0])
        cv2.imwrite('demo__ref_mask2.jpg', draw[1])

        anos = [(a['bbox'], a['category_id'])
                for a in data['annotations'] if a['image_id'] == img['id']]
        gt_boxes = [a[0] for a in anos]
        gt_labels = [a[1] for a in anos]
        # gt = vis(np.array(image),{'boxes':box_ops.box_xywh_to_xyxy(torch.as_tensor(gt_boxes)),'labels':gt_labels,'scores':[1 for a in anos]})
        # print('gt : ',[CATEGORIES[i] for i in gt_labels])

        tp, fp, fn = evaluate_result(gt_boxes, gt_labels, result)
        # tp,fp,fn = [],[],[]

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        # path = ''
        if check:
            if len(fp) == 0 and len(fn) == 0:
                path = out_ens_path+'/check/tp'
            elif len(fp) != 0:
                path = out_ens_path+'/check/fp'
            else:
                path = out_ens_path+'/check/fn'
        else:
            if len(fp) == 0 and len(fn) == 0:
                path = out_ens_path+'/tp'
            elif len(fp) != 0:
                path = out_ens_path+'/fp'
            else:
                path = out_ens_path+'/fp'

        # draw = np.hstack((draw,gt))
        pre = total_tp/(total_tp+total_fp+1e-12)
        rec = total_tp/(total_tp+total_fn+1e-12)
        f1 = 2*pre*rec/(pre+rec+1e-12)
        print('tp : ', total_tp, ' fp : ', total_fp, ' fn : ', total_fn, 'precision : ', pre,
              ' recall : ', rec, ' f1 :', f1, ' matching :', MATCHING_BOX, ' total box :', TOTAL_BOX)
        cv2.imwrite(path+'/'+file_name, draw[0])
        # cv2.imwrite('demo_mask.jpg',draw)

        # print('tp : ',[CATEGORIES[result['labels'][i]] for i in tp])
        # print('fp : ',[CATEGORIES[result['labels'][i]] for i in fp])
        # print('fn : ',[CATEGORIES[gt_labels[i]] for i in fn])

        # break


if __name__ == '__main__':
    main()
