#!/usr/bin/python3

import cv2
import json
import flask
import flask_socketio
import glob
import io
import numpy as np
import os
import pandas
import sqlite3
from werkzeug.datastructures import ImmutableOrderedMultiDict
from collections import OrderedDict

import serializer
import helper
from action import Action


class MyRequest(flask.Request):
    parameter_storage_class = ImmutableOrderedMultiDict


class MyFlask(flask.Flask):
    request_class = MyRequest


app = MyFlask(__name__)
socketio = flask_socketio.SocketIO(app)


@app.route('/api/database-list')
def api_database_list():
    database_list = glob.glob(os.path.expanduser('~/Documents/data/**/*.db'))
    return flask.jsonify(database_list)


@app.route('/api/overview')
def api_overview():
    database = os.path.expanduser(flask.request.args.get('database'))
    if not os.path.isfile(database):
        return flask.abort(404)

    with sqlite3.connect(database) as conn:
        args = flask.request.args
        if 'pushing' in database:
            sql = "select id, reward from measurement where (? = '-1' or reward < ?) and (? = '' OR id like ?);"
            params = (args.get('reward'), args.get('reward'), args.get('id'), '%' + args.get('id') + '%')
        else:
            sql = "select id, reward from measurement where (? = '-1' or reward = ?) and (final_d >= ? and final_d <= ?) and (? = '' OR id like ?);"
            params = (args.get('reward'), args.get('reward'), args.get('final_d_lower'), args.get('final_d_upper'), args.get('id'), '%' + args.get('id') + '%')
        data = pandas.read_sql_query(sql, conn, params=params)
        return app.response_class(response=data.to_json(orient='records'), mimetype='application/json')


@app.route('/api/stats')
def api_stats():
    database = os.path.expanduser(flask.request.args.get('database'))
    if not os.path.isfile(database):
        return flask.abort(404)

    with sqlite3.connect(database) as conn:
        data = pandas.read_sql_query('select * from measurement;', conn)
        stats = helper.getStats(data)
        return flask.jsonify(stats)


@app.route('/api/action/<id>')
def api_action(id):
    database = os.path.expanduser(flask.request.args.get('database'))
    with sqlite3.connect(database) as conn:
        data = pandas.read_sql_query('select * from measurement where id = ?;', conn, params=[id])
        if data.empty:
            return flask.abort(404)

        action = json.loads(data.iloc[0].to_json())
        action['images'] = glob.glob(os.path.dirname(database) + '/measurement/image-' + id + '-*.png')
        return flask.jsonify(action)


@app.route('/api/image/<id>')
def api_image(id):
    def send_image(image):
        _, image_encoded = cv2.imencode('.jpg', image)
        return flask.send_file(io.BytesIO(image_encoded), mimetype='image/jpeg')

    database = os.path.expanduser(flask.request.args.get('database'))
    suffix = flask.request.args.get('suffix', default='v')
    image = cv2.imread(os.path.dirname(database) + '/measurement/image-{}-{}.png'.format(id, suffix))
    if image is None:
        empty = np.zeros((480, 752, 1))
        cv2.putText(empty, '?', (310, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, 100, thickness=6)
        return send_image(empty)

    with sqlite3.connect(database) as conn:
        action = Action()
        data = pandas.read_sql_query('select * from measurement where id = ?;', conn, params=[id])
        if not data.empty:
            try:
                action.fromPandas(data.iloc[0])
                helper.drawPose(image, action)
            except:
                pass
        elif flask.request.args.get('pose'):
            action.fromJson(flask.request.args.get('pose'))
            helper.drawPose(image, action)

        if int(flask.request.args.get('bin', default=0)):
            helper.drawAroundBin(image, draw_lines=True)

    return send_image(image)


@app.route('/api/upload-image', methods=['POST'])
def api_upload_image():
    database = os.path.expanduser(flask.request.args.get('database'))
    id = flask.request.args.get('id')
    suffix = flask.request.args.get('suffix', default='v')
    filepath = os.path.dirname(database) + '/measurement/image-{}-{}.png'.format(id, suffix)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    image_buffer = np.fromstring(flask.request.data, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_UNCHANGED) # , cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(filepath, image)
    return flask.jsonify(success=True)


@app.route('/api/new-result', methods=['POST'])
def api_new_result():
    database = os.path.expanduser(flask.request.args.get('database'))
    create_table = not os.path.isfile(database)
    with sqlite3.connect(database) as conn:
        c = conn.cursor()

        data = json.loads(flask.request.form.get('json'), object_pairs_hook=OrderedDict)['result']
        fields, field_types, values = serializer.get_field_list(data)

        if create_table:
            print('Create new table measurement')
            field_list = ','.join(field_types + ['primary key(id)'])
            print(field_list)
            c.execute('create table if not exists measurement (' + field_list + ')')

        # Get list of splitted values
        field_values = dict(zip(fields, values))
        field_values['database'] = database
        insert_list = ','.join(map(lambda f: ':' + f, fields))
        c.execute('insert into measurement values (' + insert_list + ')', field_values)

        conn.commit()
        socketio.emit('new-result', field_values)
        return flask.jsonify(success=True)


@app.route('/api/new-attempt', methods=['POST'])
def api_new_attempt():
    data = json.loads(flask.request.form.get('json'))
    data['database'] = os.path.expanduser(flask.request.args.get('database'))
    socketio.emit('new-attempt', data)
    return flask.jsonify(success=True)


@app.route('/api/delete/<id>', methods=['POST'])
def api_delete(id):
    database = os.path.expanduser(flask.request.form.get('database'))
    with sqlite3.connect(database) as conn:
        c = conn.cursor()
        c.execute('delete from measurement where id=?', [id])
        conn.commit()
        return flask.jsonify(success=True)


@app.route('/')
def index():
    return flask.render_template('overview.html')


@app.route('/live')
def live():
    return flask.render_template('live.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080)
