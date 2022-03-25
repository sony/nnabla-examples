# -*- coding: utf-8 -*-
import nnabla as nn
from collections import OrderedDict


class GetFunctionFromInput:
    def __init__(self, _y, func_type_list=['Convolution', 'Deconvolution']):
        # [parameters]
        self.func_type_list = func_type_list
        self.no_skip_list = [
            'Affine',
            'DepthwiseConvolution',
            'DepthwiseDeconvolution'
        ]
        # [variables]
        self._func_count_outputs = OrderedDict()
        self._func_count_inputs = OrderedDict()
        self.functions = OrderedDict()
        # [process]
        self.GV_class = GetVariablesOnGraph(_y)
        self.core()

    def core(self):
        for key in self.GV_class.coef_dict_on_graph:
            var = self.GV_class.coef_dict_on_graph[key]
            self.functions[key] = self.process_one(var)

    def process_one(self, _var):
        out_list = []
        f = _var.parent
        if f is not None and f.info.type_name in self.func_type_list:
            out_list.append(f)
        elif len(_var.function_references) > 0:
            for f_next in _var.function_references:
                if f_next.info.type_name not in self.no_skip_list:
                    out_list += self.process_one(f_next.outputs[0])
        return out_list


class GetNearestFunction:
    def __init__(self, _base_func_list, _need_func_list=['BatchNormalization'], _pass_func_list=['ReLU', 'ReLU6'],
                 _get_type='prev'):
        # inputs
        self.base_func_list = _base_func_list
        self.need_func_list = _need_func_list
        self.pass_func_list = _pass_func_list
        self.get_type = _get_type
        # outputs
        self.variables = OrderedDict()
        # others
        self.func_count_dict = OrderedDict()

    def __call__(self, f):
        func_name = f.info.type_name
        if func_name not in self.func_count_dict:
            self.func_count_dict[func_name] = 1
            key = func_name + '_Output'
        else:
            self.func_count_dict[func_name] += 1
            key = func_name + \
                '_{}_Output'.format(self.func_count_dict[func_name])
        if func_name in self.base_func_list:
            self.variables[key] = OrderedDict()
            self.variables[key][func_name] = f.outputs[0]
            self.variables[key][self.get_type] = []
            self._get_need_func(f, key)

    def _get_need_func(self, f, _key):
        func_list = [f]
        first_switch = True
        while True:
            if len(func_list) == 0:
                break
            else:
                func_list = self._get_need_func_set(
                    func_list, _key, first_switch)
            first_switch = False

    def _get_need_func_set(self, f_list, _key, _first_switch):
        next_func_list = []
        for f_one in f_list:
            end_flag = self._get_need_func_one(f_one, _key, _first_switch)
            if not end_flag:
                next_func_list += self._get_next_func_list(f_one)
        return next_func_list

    def _get_need_func_one(self, f_one, _key, _first_switch):
        end_flag = True
        if type(f_one) != type(None):
            var = f_one.outputs[0]
            if f_one.info.type_name in self.need_func_list and not _first_switch:
                if var not in self.variables[_key][self.get_type]:
                    self.variables[_key][self.get_type].append(var)
            elif f_one.info.type_name in self.pass_func_list or _first_switch:
                end_flag = False
        return end_flag

    def _get_next_func_list(self, f):
        if self.get_type == 'prev':
            return self._get_need_func_prev(f)
        elif self.get_type == 'next':
            return self._get_need_func_next(f)
        else:
            print('[error] get_nearest_func.get_next_func_list()')
            print('self.get_type is unexpected value.')
            print('self.get_type = {}'.format(self.get_type))
            exit(0)

    @staticmethod
    def _get_need_func_prev(f):
        func_prev_list = []
        for in_var in f.inputs:
            if in_var.parent is not None:
                func_prev_list.append(in_var.parent)
        return func_prev_list

    @staticmethod
    def _get_need_func_next(f):
        return f.outputs[0].function_references


class GetVariablesOnGraph:
    def __init__(self, _y):
        # [variables]
        self.variables = OrderedDict()
        self.coef_dict_on_graph = OrderedDict()
        self._func_count_outputs = OrderedDict()
        self._func_count_inputs = OrderedDict()
        self.coef_dict_source = nn.get_parameters(grad_only=False)
        self.coef_names_by_value_dict = {
            v: k for k, v in self.coef_dict_source.items()}
        # [preprocess]
        _y.visit(lambda f: self._name_variables(f, _type='Output'))
        _y.visit(lambda f: self._name_variables(f, _type='Input'))
        _y.visit(self._get_coef_dict_on_graph)

    def _name_variables(self, f, _type):
        if _type in ['Input', 'Output']:
            if _type == 'Input':
                count_dict = self._func_count_inputs
                var_list = f.inputs
            elif _type == 'Output':
                count_dict = self._func_count_outputs
                var_list = f.outputs
            for var in var_list:
                if var.name == '':
                    func_name = f.info.type_name
                    if func_name in count_dict:
                        count_dict[func_name] += 1
                        var_name = func_name + \
                            '_{}_{}'.format(count_dict[func_name], _type)
                    else:
                        count_dict[func_name] = 1
                        var_name = func_name + '_{}'.format(_type)
                    var.name = var_name
                if _type == 'Output':
                    self.variables[var.name] = var
        else:
            print('[GetVariablesOnGraph] Error in loss.py')
            print('f.info.type_name = {}'.format(f.info.type_name))

    def _get_coef_dict_on_graph(self, f):
        for in_var in f.inputs:
            if in_var in self.coef_names_by_value_dict:
                in_var.name = self.coef_names_by_value_dict[in_var]
                self.coef_dict_on_graph[in_var.name] = in_var
