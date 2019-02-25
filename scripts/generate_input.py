#!/usr/bin/python3

import cv2
import os
import sqlite3
import hashlib
import pandas as pd

import helper
from action import Action


class GenerateInput:
    percent_test_set = 0.2
    border_color = helper.background_color

    def __init__(self, input_files, test_files=None, output_folder=None, train_output_file=None, test_output_file=None):
        # Accept either string or list of string as input filenames
        self.input_files = [input_files] if isinstance(input_files, str) else input_files
        self.input_files = [os.path.expanduser(f) for f in self.input_files]
        self.test_files = [test_files] if isinstance(test_files, str) else test_files
        self.image_input_folder = '/measurement/'

        self.output_directory = os.path.dirname(self.input_files[0]) + ('/' + output_folder if output_folder else '/')
        self.train_output_filename = train_output_file if train_output_file else self.output_directory + 'train.csv'
        self.test_output_filename = test_output_file if test_output_file else self.output_directory + 'test.csv'
        self.image_output_directory = self.output_directory + 'input-32/'
        self.makeDir(self.image_output_directory)
        self.model_directory = os.path.dirname(self.input_files[0]) + '/models/'

    @staticmethod
    def binaryDecision(string, p):
        return (float(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 2**16) / 2**16 < p)

    @staticmethod
    def makeDir(directory):
        os.makedirs(directory, exist_ok=True)

    def writeFile(self, file, array, shuffle=False, norm_weight=True):
        data = pd.DataFrame(array)
        if norm_weight:
            data.weights *= 1.0 / (data.weights).mean()
        if shuffle:
            data = data.sample(frac=1)
        data.to_csv(file, mode='w', index=False)

    def checkImage(self, id, suffix=''):
        return os.path.isfile(self.image_output_directory + 'image-' + id + suffix + '.png')

    def writeImage(self, id, image, suffix=''):
        cv2.imwrite(self.image_output_directory + 'image-' + id + suffix + '.png', image)

    def getImage(self, row, suffix='ed-v', draw_bin=True):
        image_path = row.dir + self.image_input_folder + 'image-{}-{}.png'.format(row.id, suffix)
        overview_image = cv2.imread(image_path, 0)
        if overview_image is None:  # If file doesn't exist
            print('File not found: {}'.format(image_path))
        if draw_bin:
            helper.drawAroundBin(overview_image, color=self.border_color)
        return overview_image

    def getAreaImage(self, image, row, size_input, size_cropped, size_output, pose=None, border_color=None):
        action = Action()
        action.fromPandas(row)

        border_color = border_color if border_color else self.border_color
        area_image = helper.getAreaOfInterest(image, action, size_cropped, border_color=border_color)
        return cv2.resize(area_image, size_output)

    def calculateWeight(self, reward, mean_reward, did_grasp_weight):
        return did_grasp_weight / mean_reward if reward == 1 else (1.0 - did_grasp_weight) / (1.0 - mean_reward)
        # row['weights'] *= 1 - abs(row.didGrasp - row.rewardPrediction)

    def generateInput(self, params):
        frames_split = []
        for file in self.input_files:
            with sqlite3.connect(file) as conn:
                frame_data = pd.read_sql('select * from measurement;', conn)
                frame_data['dir'] = os.path.dirname(file)
                frames_split.append(frame_data)
        data_split = pd.concat(frames_split, sort=True, ignore_index=True)

        if self.test_files:
            frames_test = []
            for file in self.test_files:
                with sqlite3.connect(file) as conn:
                    frame_data = pd.read_sql('select * from measurement;', conn)
                    frame_data['dir'] = os.path.dirname(file)
                    frames_test.append(frame_data)
            data_test = pd.concat(frames_test, sort=True, ignore_index=True)

        # Join predictions
        # pred = pd.read_csv(self.directory + 'predictions.csv', index_col='id').groupby(level=0).last()
        # pred = pred.groupby(level=0).last()  # Make IDs unique
        # data = data.merge(pred, how='left', left_index=True, right_index=True, suffixes=['_old', ''])

        params['ratio'] = data_split.reward.mean()
        print('Reward Mean: {:.3}'.format(params['ratio']))

        training_data = []
        test_data = []

        self.createDataWrapper(data_split, params)
        if self.test_files:
            self.createDataWrapper(data_test, params)

        for index, row in data_split.iterrows():
            is_in_test_data = self.binaryDecision(row.id, p=self.percent_test_set)
            add_to_data = test_data if is_in_test_data else training_data
            for row_append in self.createRow(row, params, single_image=is_in_test_data):
                add_to_data.append(row_append)
            if index % 1000 == 0 and index > 0:
                print('Progress: {} of {}'.format(index, len(data_split)))

        if self.test_files:
            for index, row in data_test.iterrows():
                for row_append in self.createRow(row, params, single_image=is_in_test_data):
                    test_data.append(row_append)

        self.writeFile(self.train_output_filename, training_data, shuffle=True)
        self.writeFile(self.test_output_filename, test_data, shuffle=True)
        print('Written training data to {}'.format(self.train_output_filename))
        print('Written test data to {}'.format(self.test_output_filename))

    def createDataWrapper(self, data, params):
        data['image_path'] = self.image_output_directory + 'image-' + data.id + '.png'
        self.createData(data)

    def createData(self, data):
        pass

    def createRow(self, row, params, single_image):
        pass
