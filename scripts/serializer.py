#!/usr/bin/python3


def get_field_list(d, prefix='', ignore=[]):
    fields = []
    field_types = []
    values = []
    for k, v in d.items():
        name = prefix + '_' + k if prefix else k
        if isinstance(v, dict):
            new_fields, new_field_types, new_values = get_field_list(v, name, ignore)
            fields += new_fields
            field_types += new_field_types
            values += new_values
        elif k not in ignore:
            t = {bool: 'integer', float: 'real', int: 'integer', str: 'text'}[type(v)]
            fields += [name]
            field_types += [name + ' ' + t]
            values += [int(v) if type(v) is bool else v]
    return fields, field_types, values
