import h5py
import os.path as osp


class SpatialFeatureLoader:
    def __init__(self, feature_dir):
        h5_paths = {split: osp.join(feature_dir, '%s.h5' % split)
                    for split in ('train', 'val', 'test')}
        self.h5_files = {
            split: h5py.File(path, 'r') for split, path in h5_paths.items()}

    def __del__(self):
        for f in self.h5_files.values():
            f.close()

    def load_feature(self, imageId):
        split, idx = imageId.split('_')
        idx = int(idx)
        return self.h5_files[split]['features'][idx]
