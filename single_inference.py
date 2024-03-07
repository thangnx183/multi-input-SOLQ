import json
import numpy as np
from pathlib import Path
import cv2
from pair_solq_model import Pair_SOLQ_model
import torch
from util import box_ops
from scipy.optimize import linear_sum_assignment
import os

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
path = Path('coco_data/carpart-lr')

data = json.load(open(path/'annotations_20222'/'valid.json'))
model = Pair_SOLQ_model(
        'exps-optimal-scheduler-3-e2e-opt-mask/phase-1-single-mode/checkpoint.pth', 0.3, 'cuda:0')

# checkpoints/video-cp-side-fast-solq-48.pth
# tp :  39711  fp :  19643  fn :  13185 precision :  0.6690534757556357  recall :  0.7507372958257713  f1 : 0.7075456570150918
# exps-optimal-scheduler/cp-side-50-cates-fdct-phase-4/checkpoint.pth
# tp :  43873  fp :  14778  fn :  9023 precision :  0.748034986615744  recall :  0.8294199939503932  f1 : 0.7866280581274654

def vis(image, result,fp):
    draw_image = image.copy()
    # print('1')
    draw_image = cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR)

    for idx,(b, l, c, m) in enumerate(zip(result['bboxes'], result['labels'], result['scores'], result['masks'])):
        if idx not in fp:
            clr = [255, 0, 0]
        else:
            clr = [0,255,255]
        # print(idx,l, c)
        # if CATEGORIES[l] == 'hli+lf':
        #     clr = [255,255,0]
        # elif CATEGORIES[l] == 'fbu+f':
        #     clr = [2550,0,255]

        # print(CATEGORIES[l],c)
        b = list(map(int, b))
        cv2.rectangle(draw_image, tuple(b[:2]), tuple(
            b[2:]), (102, 220, 225), thickness=2)

        if isinstance(l, str):
            cv2.putText(draw_image, l+'|'+str(round(c, 2)),
                        (b[0], b[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1)
        else:
            cv2.putText(draw_image, CATEGORIES[l]+'|'+str(round(
                c, 2)), (b[0]//2+b[2]//2, b[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 1)

        # print(m.shape)
        m = np.array(m).astype(np.uint8)
        _,cons, _ = cv2.findContours(
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


def evaluate_result(gt_boxes, gt_labels, result):
    # print('pred  : ',[CATEGORIES[i] for i in result['labels']])
    # print('gt : ',[CATEGORIES[i] for i in gt_labels])

    pred_boxes = torch.as_tensor(result['bboxes'])

    gt_boxes = box_ops.box_xywh_to_xyxy(torch.as_tensor(gt_boxes))
    # print('gt ', gt_boxes)

    if len(pred_boxes) == 0:
        return [], [], [i for i in range(len(gt_labels))]
    ious, _ = box_ops.box_iou(pred_boxes, gt_boxes)

    gt_labels = torch.as_tensor(gt_labels)
    # print(gt_labels, result['labels'])
    match_label = torch.stack(
        [gt_labels == i for i in result['labels']]).type(torch.LongTensor)

    # print(match_label)
    # print(ious)
    cost = ious+match_label
    h, w = cost.shape
    row_ind, col_ind = matching(cost, pad_value=1)
    # print(ious, row_ind, col_ind)

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


total_tp, total_fp, total_fn = 0, 0, 0
out_ens_path = 'output_ensemble_side_ref/new_scheduler'
# os.system('rm -rf '+out_ens_path)
Path(out_ens_path).mkdir(parents=True, exist_ok=True)
(Path(out_ens_path)/'tp').mkdir(parents=True, exist_ok=True)
(Path(out_ens_path)/'fp').mkdir(parents=True, exist_ok=True)
(Path(out_ens_path)/'fn').mkdir(parents=True, exist_ok=True)

for idx,i in enumerate(data['images']):
    # i = data['images'][000]
    # annos = [np.array(a['segmentation']).reshape(-1, 2).astype(np.int32)
    #         for a in data['annotations'] if a['image_id'] == i['id']]
    # print(i['file_name'])
    image = cv2.imread(str(path/'images'/i['file_name']))
    # image_annos = cv2.drawContours(image.copy(), annos, -1, (255, 0, 0), 2)

    anos = [(a['bbox'], a['category_id'])
            for a in data['annotations'] if a['image_id'] == i['id']]
    gt_boxes = [a[0] for a in anos]
    gt_labels = [a[1] for a in anos]

    # cv2.imwrite('demo.jpg',image)

    out = model.inference_filename(image)
    

    tp, fp, fn = evaluate_result(gt_boxes, gt_labels, out)
    total_tp += len(tp)
    total_fp += len(fp)
    total_fn += len(fn)
    # print(fp)
    out_image = vis(image, out,fp)
    
    if len(fp) == 0 and len(fn) == 0:
        out_path = out_ens_path+'/tp'
    elif len(fp) != 0:
        out_path = out_ens_path+'/fp'
    else:
        out_path = out_ens_path+'/fp'
    
    pre = total_tp/(total_tp+total_fp+1e-12)
    rec = total_tp/(total_tp+total_fn+1e-12)
    f1 = 2*pre*rec/(pre+rec+1e-12)
    print('tp : ', total_tp, ' fp : ', total_fp, ' fn : ', total_fn, 'precision : ', pre,
        ' recall : ', rec, ' f1 :', f1)
    cv2.imwrite(out_path+'/'+i['file_name'], out_image)
    # break
