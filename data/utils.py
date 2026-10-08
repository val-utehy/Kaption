import re
import json
import os
import shutil

import torch
import torch.distributed as dist

import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url


def _run_eval_no_spice(coco_eval):
    """Run COCOEvalCap without SPICE (avoids Stanford NLP model download)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    imgIds = coco_eval.params['image_id']
    gts = {i: coco_eval.coco.imgToAnns[i] for i in imgIds}
    res = {i: coco_eval.cocoRes.imgToAnns[i] for i in imgIds}
    # Collapse embedded newlines/whitespace: METEOR talks to its Java subprocess
    # one line at a time, so a caption containing '\n' desyncs the stdin/stdout
    # protocol and corrupts every score after it.
    gts_tok = {i: [' '.join(a['caption'].split()) for a in v] for i, v in gts.items()}
    res_tok = {i: [' '.join(a['caption'].split()) for a in v] for i, v in res.items()}
    coco_eval.eval = {}
    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
               (Rouge(), "ROUGE_L"),
               (Cider(), "CIDEr")]

    if shutil.which('java'):
        from pycocoevalcap.meteor.meteor import Meteor
        scorers.insert(1, (Meteor(), "METEOR"))
    else:
        print("Skipping METEOR: 'java' not found on PATH (METEOR's scorer is a Java jar).")

    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts_tok, res_tok)
        if isinstance(method, list):
            for m, s in zip(method, score):
                coco_eval.eval[m] = s
        else:
            coco_eval.eval[method] = score


def coco2017_caption_eval(ann_file, results_file):
    """COCOEvalCap on a local COCO-2017-style annotation file.

    `ann_file` should point to e.g. captions_val2017.json — already on disk,
    so no download step. SPICE is skipped (Stanford NLP model not bundled).
    """
    assert os.path.exists(ann_file), f'COCO ann file not found: {ann_file}'
    coco = COCO(ann_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    _run_eval_no_spice(coco_eval)

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval


def flickr8k_caption_eval(ann_root, results_file, split):
    annotation_file = os.path.join(ann_root, f'flickr8k_{split}_gt.json')
    assert os.path.exists(annotation_file), \
        f'GT file not found: {annotation_file}. Initialize the dataset first.'

    coco = COCO(annotation_file)
    coco.dataset.setdefault('info', {})
    coco.dataset.setdefault('licenses', [])
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    # Skip SPICE: requires Stanford NLP models that need a separate download
    _run_eval_no_spice(coco_eval)

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval


def coco_caption_eval(coco_gt_root, results_file, split):
    """COCO Karpathy split eval. Downloads the BLIP Karpathy GT JSON if absent.

    Skips SPICE (Stanford NLP not bundled) — reports BLEU 1-4, METEOR, ROUGE,
    CIDEr. These are the metrics every captioning SOTA paper reports on the
    Karpathy test split, so results are directly comparable.
    """
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    _run_eval_no_spice(coco_eval)

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    return coco_eval