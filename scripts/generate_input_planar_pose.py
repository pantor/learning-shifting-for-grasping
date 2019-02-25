#!/usr/bin/python3

import numpy as np

import helper

from generate_input import GenerateInput


class GenerateInputPlanarPose(GenerateInput):
    def __init__(self, input_files, test_files=None, output_folder=None, train_output_file=None, test_output_file=None):
        GenerateInput.__init__(self, input_files, test_files, output_folder, train_output_file, test_output_file)

    def createData(self, data):
        data['gripper_class'] = data.action_pose_d.map(helper.getGripperClass)
        data['reward_given'] = 1
        data['weights'] = 1.0

    def createRow(self, row, params, single_image):
        if not self.checkImage(row.id) or params['force_rewrite']:
            depth_image = self.getImage(row, suffix='ed-v')
            if 'inpaint' in params and params['inpaint']:
                uncertainty = np.zeros(depth_image.shape, np.uint8)
                uncertainty[(depth_image == 0)] = 255
                output_uncertainty = self.getAreaImage(uncertainty, row, params['size_input'], params['size_cropped'], params['size_output'])
                self.writeImage(row.id, output_uncertainty, '-u')

                helper.inpaint(depth_image)

            if 'raw_images' in params and params['raw_images']:
                raw_image = self.getImage(row, suffix='er-v')
                output_raw_image = self.getAreaImage(raw_image, row, params['size_input'], params['size_cropped'], params['size_output'])
                self.writeImage(row.id, output_raw_image, suffix='-r')

            output_depth_image = self.getAreaImage(depth_image, row, params['size_input'], params['size_cropped'], params['size_output'])
            self.writeImage(row.id, output_depth_image)

        # Calculate weight
        row['weights'] = self.calculateWeight(row.reward, params['ratio'], params['did_grasp_weight'])

        if 'raw_images' in params and params['raw_images']:
            row['raw_image_path'] = self.image_output_directory + 'image-{}-r.png'.format(row.id)
        yield row


if __name__ == '__main__':
    generator = GenerateInputPlanarPose(
        '~/Documents/data/cylinder-1/cylinder-1.db'
    )
    generator.generateInput({
        'did_grasp_weight': 0.13,  # empiric value, false positive vs. false negative error
        'force_rewrite': False,
        'inpaint': False,
        'size_cropped': (200, 200),  # pxSize: 0.6, 240: 0.5
        'size_input': (752, 480),
        'size_output': (32, 32),
    })
