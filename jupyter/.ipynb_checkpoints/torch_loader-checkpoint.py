import imageio
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, file, label_fields=['reward', 'gripper_class', 'weights'], batch_size=128):
        self.load_data(file, label_fields)
        self.batch_size = batch_size
        self.batches_per_epoch = int(np.floor(len(self) / self.batch_size))
        
    def __len__(self):
        return len(self.labels)

    def load_data(self, file, label_fields):
        data = pd.read_csv(file)
        
        self.images = np.array([np.expand_dims(imageio.imread(path, pilmode='P'), axis=2) for path in data['image_path'].values]) / 255.
        self.labels = data[label_fields].values

        print('Length: {}'.format(len(data)))
        print('Reward Mean: {}'.format(data.reward.mean()))

    def nextBatch(self):
        for i in range(self.batches_per_epoch - 1):
        #for i in range(len(self)):
            low = i * self.batch_size
            up = (i + 1) * self.batch_size
            yield (self.images[low:up], self.labels[low:up])
        return

    def entireBatch(self):
        yield (self.images, self.labels)
        return
