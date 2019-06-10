import threading
import queue
import numpy as np

from util import text_processing
from util.positional_encoding import get_positional_encoding
from util.clevr_feature_loader.feature_loader import SpatialFeatureLoader
from util.boxes import bbox2feat_grid


class BatchLoaderClevr:
    def __init__(self, imdb, data_params):
        self.imdb = imdb
        self.data_params = data_params

        self.vocab_dict = text_processing.VocabDict(
            data_params['vocab_question_file'])
        self.T_encoder = data_params['T_encoder']

        # peek one example to see whether answer is in the data
        self.load_answer = ('answer' in self.imdb[0])
        # peek one example to see whether bbox is in the data
        self.load_bbox = ('bbox' in self.imdb[0])
        # the answer dict is always loaded, regardless of self.load_answer
        self.answer_dict = text_processing.VocabDict(
            data_params['vocab_answer_file'])
        if not (self.load_answer or self.load_bbox):
            print('imdb has no answer labels or bbox. Using dummy labels.\n\n'
                  '**The final accuracy will be zero (no labels provided)**\n')

        # positional encoding
        self.add_pos_enc = data_params.get('add_pos_enc', False)
        self.pos_enc_dim = data_params.get('pos_enc_dim', 0)
        assert self.pos_enc_dim % 4 == 0, \
            'positional encoding dim must be a multiply of 4'
        self.pos_enc_scale = data_params.get('pos_enc_scale', 1.)

        self.load_spatial_feature = data_params['load_spatial_feature']
        if self.load_spatial_feature:
            spatial_feature_dir = data_params['spatial_feature_dir']
            self.spatial_loader = SpatialFeatureLoader(spatial_feature_dir)
            # load one feature map to peek its size
            x = self.spatial_loader.load_feature(self.imdb[0]['imageId'])
            self.spatial_D, self.spatial_H, self.spatial_W = x.shape
            # positional encoding
            self.pos_enc = self.pos_enc_scale * get_positional_encoding(
                self.spatial_H, self.spatial_W, self.pos_enc_dim)

        if self.load_bbox:
            self.img_H = data_params['img_H']
            self.img_W = data_params['img_W']
            self.stride_H = self.img_H * 1. / self.spatial_H
            self.stride_W = self.img_W * 1. / self.spatial_W

    def load_one_batch(self, sample_ids):
        actual_batch_size = len(sample_ids)
        input_seq_batch = np.zeros(
            (actual_batch_size, self.T_encoder), np.int32)
        seq_length_batch = np.zeros(actual_batch_size, np.int32)
        if self.load_spatial_feature:
            spatial_feat_batch = np.zeros(
                (actual_batch_size, self.spatial_D, self.spatial_H,
                 self.spatial_W), np.float32)
        if self.load_bbox:
            bbox_batch = np.zeros((actual_batch_size, 4), np.float32)
            bbox_ind_batch = np.zeros(actual_batch_size, np.int32)
            bbox_offset_batch = np.zeros((actual_batch_size, 4), np.float32)

        qid_list = [None]*actual_batch_size
        qstr_list = [None]*actual_batch_size
        imageid_list = [None]*actual_batch_size
        if self.load_answer:
            answer_label_batch = np.zeros(actual_batch_size, np.int32)
        else:
            answer_label_batch = -np.ones(actual_batch_size, np.int32)
        for n in range(len(sample_ids)):
            iminfo = self.imdb[sample_ids[n]]
            question_str = iminfo['question']
            question_tokens = text_processing.tokenize_clevr(question_str)
            if len(question_tokens) > self.T_encoder:
                print('data reader: truncating question:\n\t' + question_str)
                question_tokens = question_tokens[:self.T_encoder]
            question_inds = [
                self.vocab_dict.word2idx(w) for w in question_tokens]
            seq_length = len(question_inds)
            input_seq_batch[n, :seq_length] = question_inds
            seq_length_batch[n] = seq_length
            if self.load_spatial_feature:
                feature = self.spatial_loader.load_feature(iminfo['imageId'])
                spatial_feat_batch[n:n+1] = feature
            qid_list[n] = iminfo['questionId']
            qstr_list[n] = question_str
            imageid_list[n] = iminfo['imageId']
            if self.load_answer:
                answer_idx = self.answer_dict.word2idx(iminfo['answer'])
                answer_label_batch[n] = answer_idx
            if self.load_bbox:
                bbox_batch[n] = iminfo['bbox']
                bbox_ind_batch[n], bbox_offset_batch[n] = bbox2feat_grid(
                    iminfo['bbox'], self.stride_H, self.stride_W,
                    self.spatial_H, self.spatial_W)
        batch = dict(input_seq_batch=input_seq_batch,
                     seq_length_batch=seq_length_batch,
                     answer_label_batch=answer_label_batch,
                     qid_list=qid_list, qstr_list=qstr_list,
                     imageid_list=imageid_list)
        if self.load_spatial_feature:
            # NCHW -> NHWC
            spatial_feat_batch = spatial_feat_batch.transpose((0, 2, 3, 1))
            batch['spatial_feat_batch'] = spatial_feat_batch
            if self.add_pos_enc:
                # add positional embedding to the image features
                pos_enc_tile = np.tile(
                    self.pos_enc, (len(spatial_feat_batch), 1, 1, 1))
                image_feat_batch = np.concatenate(
                     (spatial_feat_batch, pos_enc_tile), axis=-1)
            else:
                image_feat_batch = spatial_feat_batch
            N, H, W, C = image_feat_batch.shape
            image_feat_batch = image_feat_batch.reshape((N, H*W, C))
            image_valid_batch = np.ones(image_feat_batch.shape[:-1], np.bool)
        batch['image_feat_batch'] = image_feat_batch
        batch['image_valid_batch'] = image_valid_batch
        if self.load_bbox:
            batch['bbox_batch'] = bbox_batch
            batch['bbox_ind_batch'] = bbox_ind_batch
            batch['bbox_offset_batch'] = bbox_offset_batch
        return batch


class DataReader:
    def __init__(self, data_file, shuffle, max_num=0, prefetch_num=16,
                 **kwargs):
        print('Loading imdb from %s' % data_file)
        imdb = np.load(data_file, allow_pickle=True)
        print('Done')
        self.imdb = imdb
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num
        self.data_params = kwargs
        if max_num > 0 and max_num < len(self.imdb):
            print('keeping %d samples out of %d' % (max_num, len(self.imdb)))
            self.imdb = self.imdb[:max_num]

        # Vqa data loader
        self.batch_loader = BatchLoaderClevr(self.imdb, self.data_params)

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch_num)
        self.prefetch_thread = threading.Thread(
            target=_run_prefetch, args=(
                self.prefetch_queue, self.batch_loader, self.imdb,
                self.shuffle, self.data_params))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def batches(self, one_pass):
        while True:
            # Get a batch from the prefetching queue
            # if self.prefetch_queue.empty():
            #     print('data reader: waiting for IO...')
            batch, n_sample, n_epoch = self.prefetch_queue.get(block=True)
            if batch is None:
                if one_pass:
                    raise StopIteration()
                else:
                    # get the next batch
                    batch, n_sample, n_epoch = self.prefetch_queue.get(
                        block=True)
            yield (batch, n_sample, n_epoch)


def _run_prefetch(prefetch_queue, batch_loader, imdb, shuffle, data_params):
    num_samples = len(imdb)
    batch_size = data_params['batch_size']

    n_sample = 0
    n_epoch = 0
    fetch_order = np.arange(num_samples)
    while True:
        # Shuffle the sample order for every epoch
        if n_sample == 0 and shuffle:
            fetch_order = np.random.permutation(num_samples)

        # Load batch from file
        # note that len(sample_ids) <= batch_size, not necessarily equal
        sample_ids = fetch_order[n_sample:n_sample+batch_size]
        batch = batch_loader.load_one_batch(sample_ids)
        prefetch_queue.put((batch, n_sample, n_epoch), block=True)

        n_sample += len(sample_ids)
        if n_sample >= num_samples:
            n_sample = 0
            n_epoch += 1
            # Put in a None batch to indicate an epoch is over
            prefetch_queue.put((None, n_sample, n_epoch), block=True)
