import numpy as np
import json
import os


def build_imdb(image_set):
    print('building imdb %s' % image_set)
    question_file = '../clevr_locplus_dataset/refexps/clevr_ref+_%s_refexps.json'
    scene_file = '../clevr_locplus_dataset/scenes/clevr_ref+_%s_scenes.json'
    with open(question_file % image_set.replace('locplus_', '')) as f:
        questions = json.load(f)['refexps']
    with open(scene_file % image_set.replace('locplus_', '')) as f:
        scenes = json.load(f)['scenes']
    imdb = []
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        questionId = '%s_%s' % (image_set, q['refexp_index'])
        imageId = '%s_%s' % (image_set.replace('locplus_', ''), q['image_index'])
        question = q['refexp']
        image_name = q['image_filename']
        iminfo = dict(questionId=questionId,
                      imageId=imageId,
                      question=question,
                      image_name=image_name)
        # find boxes
        scene = scenes[q['image_index']]
        assert q['image_filename'] == scene['image_filename']
        obj_inds = q['program'][-1]['_output']
        if len(obj_inds) > 1:  # skip refexps with more than one target object
            continue
        obj_boxes = scene['obj_bbox']
        obj = scene['objects'][obj_inds[0]]
        bbox = obj_boxes[str(obj['idx'])]
        iminfo['bbox'] = bbox

        imdb.append(iminfo)
    return imdb


imdb_trn = build_imdb('locplus_train')
imdb_val = build_imdb('locplus_val')

os.makedirs('./imdb', exist_ok=True)
np.save('./imdb/imdb_locplus_train.npy', np.array(imdb_trn))
np.save('./imdb/imdb_locplus_val.npy', np.array(imdb_val))
