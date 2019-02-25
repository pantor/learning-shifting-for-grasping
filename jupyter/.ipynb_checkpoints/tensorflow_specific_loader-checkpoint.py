import imageio
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, file, n_pos_pos, n_pos_neg, n_neg_neg, n_pos_no_ann=0, n_neg_no_ann=0, label_fields=['reward', 'gripper_class', 'weights'], batch_size=128):
        self.load_data(file, label_fields)
        self.batch_size = batch_size
        self.batches_per_epoch = int(np.floor(len(self) / self.batch_size))

        self.n_pos_pos = n_pos_pos
        self.n_pos_neg = n_pos_neg
        self.n_pos_no_ann = n_pos_no_ann
        self.n_neg_neg = n_neg_neg
        self.n_neg_no_ann = n_neg_no_ann

    def __len__(self):
        return len(self.labels)

    def load_data(self, file, label_fields):
        data = pd.read_csv(file)

        self.images = np.array([np.expand_dims(imageio.imread(path, pilmode='P'), axis=2) for path in data['image_path'].values]) / 255.
        self.anns = np.array([np.expand_dims(imageio.imread(path, pilmode='P'), axis=2) for path in data['ann_path'].values]) / 255.
        self.labels = data[label_fields].values

        print('Length: {}'.format(len(data)))
        print('Reward Mean: {}'.format(data.reward.mean()))

    def nextBatch(self):
        for i in range(len(self) - 1):
            low = i * self.batch_size
            up = (i + 1) * self.batch_size
            yield self.createSpecific(self.images[low:up], self.anns[low:up], self.labels[low:up])
        return

    def entireBatch(self):
        yield self.createSpecific(self.images, self.anns, self.labels)
        return

    def createSpecific(self, images, anns, labels):
        _images, _anns, _labels = [], [], []

        def append(image, ann, label, reward):
            _label = label.copy()
            _label[0] = reward
            _images.append(image)
            _anns.append(ann)
            _labels.append(_label)

        for image, ann, label in zip(images, anns, labels):
            if label[0] == 1.0:
                prob = ann.flatten() / np.sum(ann) if np.sum(ann) > 0 else None
                for i in range(self.n_pos_pos):
                    random_point = np.unravel_index(np.random.choice(np.arange(ann.size), p=prob), ann.shape)
                    _ann = np.zeros(ann.shape)
                    _ann[random_point] = 1.0
                    append(image, _ann, label, reward=1.0)

                prob = np.ravel(1.0 - ann) / np.sum(1.0 - ann)
                for i in range(self.n_pos_neg):
                    random_point = np.unravel_index(np.random.choice(np.arange(ann.size), p=prob), ann.shape)
                    _ann = np.zeros(ann.shape)
                    _ann[random_point] = 1.0
                    append(image, _ann, label, reward=0.0)

                for i in range(self.n_pos_no_ann):
                    _ann = np.zeros(ann.shape)
                    append(image, _ann, label, reward=0.0)
            else:
                for i in range(self.n_neg_neg):
                    random_point = np.unravel_index(np.random.choice(np.arange(ann.size)), ann.shape)
                    _ann = np.zeros(ann.shape)
                    _ann[random_point] = 1.0
                    append(image, _ann, label, reward=0.0)

                for i in range(self.n_neg_no_ann):
                    _ann = np.zeros(ann.shape)
                    append(image, _ann, label, reward=0.0)

        return _images, _anns, _labels
