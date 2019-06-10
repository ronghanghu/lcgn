import json
import sys; sys.path.append('../../util/')  # NoQA
from text_processing import tokenize_clevr
from collections import defaultdict


# We collect vocabulary and answers from the (unbalanced) full training set
question_files = [
    '../clevr_locplus_dataset/refexps/clevr_ref+_train_refexps.json',
    '../clevr_locplus_dataset/refexps/clevr_ref+_val_refexps.json'
]
vocab_file = './vocabulary_clevr_locplus_drew.txt'


vocab_count = defaultdict(int)
for question_file in question_files:
    print('loading ' + question_file)
    with open(question_file) as f:
        questions = json.load(f)['refexps']
    for q in questions:
        words = tokenize_clevr(q['refexp'])
        for w in words:
            vocab_count[w] += 1

sorted_vocab = ['<pad>', '<unk>', '<start>', '<end>'] + sorted(vocab_count)
with open(vocab_file, 'w') as f:
    for w in sorted_vocab:
        f.write(w+'\n')
