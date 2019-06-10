import numpy as np
import json
import os


def build_imdb(image_set):
    print('building imdb %s' % image_set)
    question_file = '../clevr_dataset/questions/CLEVR_%s_questions.json'
    with open(question_file % image_set) as f:
        questions = json.load(f)['questions']
    imdb = []
    for n_q, q in enumerate(questions):
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        questionId = '%s_%s' % (image_set, q['question_index'])
        imageId = '%s_%s' % (image_set, q['image_index'])
        question = q['question']
        image_name = q['image_filename']
        iminfo = dict(questionId=questionId,
                      imageId=imageId,
                      question=question,
                      image_name=image_name)
        if 'answer' in q:
            iminfo['answer'] = q['answer']

        imdb.append(iminfo)
    return imdb


imdb_trn = build_imdb('train')
imdb_val = build_imdb('val')
imdb_tst = build_imdb('test')

os.makedirs('./imdb', exist_ok=True)
np.save('./imdb/imdb_train.npy', np.array(imdb_trn))
np.save('./imdb/imdb_val.npy', np.array(imdb_val))
np.save('./imdb/imdb_test.npy', np.array(imdb_tst))
