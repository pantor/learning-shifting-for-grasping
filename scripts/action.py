#!/usr/bin/python3

import json


class Action:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.d = 0.0

        self.found = False
        self.prob = 0.0
        self.probstd = 0.0
        self.method = ''

        self.reward = 0
        self.collision = False

    def fromPandas(self, data):
        self.x = data['action_pose_x']
        self.y = data['action_pose_y']
        self.z = data['action_pose_z']
        self.a = data['action_pose_a']
        self.b = data['action_pose_b']
        self.c = data['action_pose_c']
        self.d = data['action_pose_d']

        self.reward = data.reward
        self.collision = data.collision

    def fromJson(self, str):
        data = json.loads(str)
        self.x = float(data['pose']['x'])
        self.y = float(data['pose']['y'])
        self.z = float(data['pose']['z'])
        self.a = float(data['pose']['a'])
        self.b = float(data['pose']['b'])
        self.c = float(data['pose']['c'])
        self.d = float(data['pose']['d'])

    def toJson(self):
        return json.dumps({
            'action': {
                'pose': {
                    'x': float(self.x),
                    'y': float(self.y),
                    'z': float(self.z),
                    'a': float(self.a),
                    'b': float(self.b),
                    'c': float(self.c),
                    'd': float(self.d),
                },
                'found': self.found,
                'prob': float(self.prob),
                'probstd': float(self.probstd),
                'method': self.method,
            }
        })

    def __repr__(self):
        return 'x: {:.3f}, y: {:.3f}, z: {:.3f}, a: {:.3f}, b: {:.3f}, c: {:.3f}, d: {:.2f}, p: {:04.2f}'.format(self.x, self.y, self.z, self.a, self.b, self.c, self.d, self.prob)
