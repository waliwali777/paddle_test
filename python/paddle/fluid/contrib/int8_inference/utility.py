#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
import paddle.fluid.core as core
import numpy as np
import math
import os
import paddle.fluid as fluid


class Calibrator(object):
    '''
    The calibrator class transforms the program and updates the calculated scale into it.
    '''
    non_conv_int8_op_type = ("pool2d")
    supported_int8_op_type = ("conv2d", "pool2d")
    not_supported_int8_op_type = ("elementwise_add", "concat")
    u8_max = 255
    s8_max = 127

    def __init__(self, *args, **kwargs):
        self.program = kwargs['program']
        self.iterations = kwargs['iterations']
        self.pretrained_model = kwargs['pretrained_model']
        self.debug = kwargs['debug']
        self.algo = kwargs['algo']

        self._conv_input_var_name = []
        self._conv_output_var_name = []
        self._weights_var_name = []
        self._residual_input_var_name = []
        self._int8_output_var_op_index_dict = {}
        self._conv_op_index = [
            index for index, value in enumerate(self.program.global_block().ops)
            if value.type == 'conv2d'
        ]

        self._var_max_value_map = {}
        self._var_max_range = {}
        self._u8_output_var = []
        self._s8_output_var = []
        self._persistable_vars = []

    def generate_sampling_program(self):
        self.__init_analysis()
        self.__generate_output_program()

    def generate_quantized_data(self, sampling_data):
        self.__sampling(sampling_data)
        self.__save_scale()
        self.__update_program()
        self.__update_output_program_attr()
        self.__display_debug()

    def __display_debug(self):
        if self.debug:
            self.__dot(self._output_program)
            print(self._output_program)

    def __get_max_range_by_var_name(self, program, var_name):
        """
        Check the specified variable was generatd from Relu layer or not.
        """
        search_end_index = 0
        input_index_name = {}
        output_index_name = {}
        ops_type = []
        found_var = False
        first_conv_op_index = -1
        first_conv_op_flag = False
        for index, op in enumerate(program.current_block().ops):
            ops_type.append(op.type)
            if op.type == 'conv2d' and not first_conv_op_flag:
                first_conv_op_index = index
                first_conv_op_flag = True
            input_index_name[index] = op.input_arg_names

            output_index_name[index] = op.output_arg_names
            if var_name in op.output_arg_names:
                search_end_index = index
                found_var = True

        # analysis
        while search_end_index >= 0:
            if ops_type[search_end_index] == "relu":
                return Calibrator.u8_max

            input_name = input_index_name[search_end_index][0]
            found_ancestor = False

            for i in output_index_name.keys():
                if input_name in output_index_name[i]:
                    search_end_index = i
                    found_ancestor = True
                    break

            if not found_ancestor:  # Dangling var
                return Calibrator.s8_max
            elif search_end_index == first_conv_op_index:  # first conv
                if program.current_block().ops[i].has_attr(
                        'fuse_relu') and program.current_block().ops[i].attr(
                            'fuse_relu'):
                    return Calibrator.u8_max
                else:
                    return Calibrator.s8_max
            elif ops_type[search_end_index] != 'conv2d':
                continue
            else:
                if program.current_block().ops[i].has_attr(
                        'fuse_relu') and program.current_block().ops[i].attr(
                            'fuse_relu'):
                    return Calibrator.u8_max
                else:
                    return Calibrator.s8_max
        return Calibrator.s8_max

    def __check_op_type_with_specified_var_as_input(self,
                                                    program,
                                                    var_name,
                                                    start_index=0):
        '''
        Check all the type of ops that use the specified variable as the input.
        If one of those op is not int8-enabled, return False.
        '''
        op_type_list = [
            op.type for op in program.current_block().ops[start_index:]
            if var_name in op.input_arg_names
        ]
        for i in op_type_list:
            if not i in Calibrator.supported_int8_op_type:
                return False
        return True

    def __check_var_source_dt(self, var_name):
        '''
        Check the specified variable is the output of int8 op or not.
        If true, return the original op index.
        If false, return -1
        '''
        return self._int8_output_var_op_index_dict[
            var_name] if var_name in self._int8_output_var_op_index_dict else -1

    def __update_int8_output_var_op_index_dict(self, index, var_name=None):
        '''
        Update the int8_output_variable/op_index dictionary
        '''
        for k, v in self._int8_output_var_op_index_dict.items():
            if v >= index:
                self._int8_output_var_op_index_dict[k] = v + 1
        if var_name:
            self._int8_output_var_op_index_dict[var_name] = index

    def __update_program(self):
        '''
        Update the program with the quantize/dequantize op insertion.
        '''
        quantize_index, dequantize_index = self.__get_quantize_dequantize_combination(
            self._output_program)
        inserted_op_length = 0
        calc_max_func = self.__get_optimal_scaling_factor if self.algo == "KL" else np.max
        insert_op_collection = sorted(quantize_index + dequantize_index)

        for index in insert_op_collection:
            if index in quantize_index:

                quantize_tmp = self._output_program.current_block().create_var(
                    name="quantize_{}_tmp".format(index),
                    dtype=core.VarDesc.VarType.UINT8)
                original_out_name = self._output_program.current_block().ops[
                    index + inserted_op_length].output_names[0]
                original_out = self._output_program.current_block().ops[
                    index + inserted_op_length].output(original_out_name)[0]

                op = self._output_program.current_block()._insert_op(
                    index=index + inserted_op_length + 1,
                    type="quantize",
                    inputs={"Input": original_out},
                    outputs={"Output": quantize_tmp}, )

                op._set_attr("data_format", "MKLDNNLAYOUT")
                op._set_attr("use_mkldnn", 1)
                op._set_attr(
                    "Scale", self._var_max_range[original_out] /
                    calc_max_func(self._var_max_value_map[original_out]))

                if self.__get_max_range_by_var_name(
                        self._output_program,
                        original_out) == Calibrator.s8_max:
                    op._set_attr("is_negative_input", 1)

                self.__update_int8_output_var_op_index_dict(
                    index + inserted_op_length, "quantize_{}_tmp".format(index))

                inserted_op_length += 1
                for op in self._output_program.current_block().ops[
                        index + inserted_op_length:]:
                    for j in op.input_names:
                        if op.input(j) and op.input(
                                j
                        )[0] == original_out and op.type in Calibrator.supported_int8_op_type:
                            op.desc.set_input(j,
                                              ["{}".format(quantize_tmp.name)])
            else:
                start_index = index + inserted_op_length
                dequantize_tmp_var = self._output_program.current_block(
                ).create_var(
                    name="dequantize_{}_tmp".format(index + 1),
                    dtype="float32", )
                original_out_var = None

                for original_input in self._output_program.current_block().ops[
                        start_index].input_arg_names:
                    index_res = self.__get_op_index_by_output_var(
                        self._output_program, original_input)
                    if index_res != -1:
                        original_out_var = original_input
                        break

                if original_out_var:
                    op = self._output_program.current_block()._insert_op(
                        index=start_index,
                        type="dequantize",
                        inputs={"Input": original_out_var},
                        outputs={"Output": dequantize_tmp_var})
                    op._set_attr("data_format", "MKLDNNLAYOUT")
                    op._set_attr("use_mkldnn", 1)
                    op._set_attr("Scale", self._var_max_range[original_out_var]
                                 / calc_max_func(self._var_max_value_map[
                                     original_out_var]))

                    for op_index in range(
                            start_index + 1,
                            len(self._output_program.current_block().ops)):
                        for j in self._output_program.current_block().ops[
                                op_index].input_names:
                            if len(self._output_program.current_block(
                            ).ops[op_index].input(
                                    j)) and self._output_program.current_block(
                                    ).ops[op_index].input(j)[
                                        0] == original_out_var:
                                self._output_program.current_block(
                                ).ops[op_index].desc.set_input(
                                    j, ["{}".format(dequantize_tmp_var.name)])

                    inserted_op_length += 1

                    op._set_attr("data_format", "MKLDNNLAYOUT")
                    op._set_attr("use_mkldnn", 1)
                    self.__update_int8_output_var_op_index_dict(start_index)

    def __update_output_program_attr(self):
        for i in self._output_program.list_vars():
            if i.name in self._persistable_vars:
                i.persistable = False
                os.system("rm -rf {}/{}".format(self.pretrained_model, i.name))

        for i in self._u8_output_var:
            self._output_program.current_block().var(i).desc.set_dtype(
                core.VarDesc.VarType.UINT8)

        for i in self._s8_output_var:
            self._output_program.current_block().var(i).desc.set_dtype(
                core.VarDesc.VarType.INT8)

    @property
    def sampling_program(self):
        return self._output_program

    @property
    def sampling_vars(self):
        return self._weights_var_name + self._conv_input_var_name + self._conv_output_var_name + self._residual_input_var_name

    def _is_close(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def __generate_output_program(self):
        for i in self.program.list_vars():
            if not i.persistable and i.name in self._weights_var_name + self._conv_input_var_name + self._conv_output_var_name + self._residual_input_var_name:
                i.persistable = True
                self._persistable_vars.append(i.name)

        self._output_program = self.program.clone()

    def __save_scale(self):
        '''
        Update the convolution scale information.
        '''
        func = self.__get_optimal_scaling_factor if self.algo == 'KL' else np.max

        for i in self._conv_op_index[1:]:
            weights_var_name = self.program.current_block().ops[i].input(
                'Filter')[0]
            input_var_name = self.program.current_block().ops[i].input('Input')[
                0]
            output_var_name = self.program.current_block().ops[i].output(
                'Output')[0]
            self._output_program.current_block().ops[i]._set_attr(
                "Scale_weights", self._var_max_value_map[weights_var_name])
            self._output_program.current_block().ops[i]._set_attr(
                "Scale_in", self._var_max_range[input_var_name] /
                func(self._var_max_value_map[input_var_name]))

            self._output_program.current_block().ops[i]._set_attr(
                "Scale_out", self._var_max_range[output_var_name] /
                func(self._var_max_value_map[output_var_name]))
            if self._output_program.current_block().ops[i].desc.input(
                    "ResidualData"):
                residual_var_name = self._output_program.current_block().ops[
                    i].desc.input("ResidualData")[0]
                self._output_program.current_block().ops[i]._set_attr(
                    "Scale_in_eltwise", self._var_max_range[residual_var_name] /
                    func(self._var_max_value_map[residual_var_name]))

    def __sampling(self, sampling_data):
        '''
        Sampling the variables data range.
        '''
        for i in self.program.list_vars():
            if i.name not in self._weights_var_name + self._conv_input_var_name + self._conv_output_var_name + self._residual_input_var_name:
                continue

            if i.name in self._weights_var_name:
                max_value = []
                data = sampling_data[i.name][0]
                for j in range(data.shape[0]):
                    if not self._is_close(float(np.max(np.abs(data[j]))), 0.0):
                        max_value.append(Calibrator.s8_max /
                                         float(np.max(np.abs(data[j]))))
                    else:
                        max_value.append(0.0)
                max_range = Calibrator.s8_max
            else:
                if i.name in self._conv_output_var_name:
                    cur_op = self.program.current_block().ops[
                        self._conv_output_var_name.index(i.name) + 2]
                    if cur_op.has_attr('fuse_relu') and cur_op.attr(
                            'fuse_relu'):
                        max_range = Calibrator.u8_max
                        self._u8_output_var.append(i.name)
                    else:
                        max_range = Calibrator.s8_max
                        self._s8_output_var.append(i.name)
                else:
                    max_range = self.__get_max_range_by_var_name(self.program,
                                                                 i.name)

                max_value = [[np.abs(np_data)]
                             for np_data in sampling_data[i.name]]

            self._var_max_range[i.name] = max_range

            self._var_max_value_map[i.name] = max_value

    def __check_force_fp32_attr_by_output_var(self, program, var_name):
        for op in program.current_block().ops:
            if op.type == "conv2d" and var_name in op.output_arg_names:
                return op.attr("force_fp32_output")
        return False

    def __get_op_index_by_output_var(self, program, var_name, start_index=0):
        '''
        Check the specified input variable is the output of the conv/pool2d
        op's output or not.

        Returns:
            The index if the variable is the output of any conv/pool2d op's
            output.
            -1 when the variable is not the output of any conv/pool2d op's 
            output.
        '''
        for index, op in enumerate(program.current_block().ops[start_index:]):
            if var_name in op.output_arg_names and op.type in Calibrator.supported_int8_op_type:
                return index
        return -1

    def __get_op_index_by_input_var(self, program, var_name, start_index=0):
        '''
        Get the op index by specified input variable.
        Returns:
            The op index if the variable is the input of this op or -1 if the 
            variable is not the input of any op. 
        '''
        for index, op in enumerate(program.current_block().ops[start_index:]):
            if var_name in op.input_arg_names:
                return index

        return -1

    def __get_quantize_dequantize_combination(self, program):
        """
        Get the quantize/dequantize op index for further inserting.
        Args:
            The program desc.
        Returns:
            Two lists contains the quantize op and dequantize op index information.
        """
        quantize_op_index = []
        dequantize_op_index = []

        if len(self._conv_op_index) < 2:
            return [], []

        for index, value in enumerate(self._conv_op_index):
            if index == 0:
                quantize_op_index.append(self._conv_op_index[index + 1] - 1)
            elif index == len(self._conv_op_index) - 1:
                output_var = program.current_block().ops[value].output(
                    "Output")[0]
                if self.__check_op_type_with_specified_var_as_input(
                        program, output_var, index):
                    dequantize_op_index.append(self._conv_op_index[index] + 2)
                else:
                    program.current_block().ops[value]._set_attr(
                        "force_fp32_output", True)

            else:
                if self._conv_op_index[index] + 1 < self._conv_op_index[index +
                                                                        1]:
                    for op_index in range(self._conv_op_index[index] + 1,
                                          self._conv_op_index[index + 1]):
                        op_type = program.current_block().ops[op_index].type
                        if op_type in "conv2d" and self._conv_op_index[
                                op_index] < self._conv_op_index[index + 1]:
                            continue
                        elif op_type in Calibrator.non_conv_int8_op_type:
                            # dequantize_op_index.append(op_index) #disable dequantize if enable force_fp32_output
                            program.current_block().ops[op_index - 1]._set_attr(
                                "force_fp32_output", True)
                            break
                        else:
                            program.current_block().ops[op_index - 1]._set_attr(
                                "force_fp32_output", True)
                            break

                    for op_index in range(self._conv_op_index[index + 1],
                                          self._conv_op_index[index], -1):
                        op_type = program.current_block().ops[op_index].type
                        op_has_int8_input = False
                        input_length = len(program.current_block().ops[op_index]
                                           .input_names)
                        for input_attr_name in program.current_block().ops[
                                op_index].input_names:
                            if program.current_block().ops[op_index].input(
                                    input_attr_name):
                                input_var_name = program.current_block().ops[
                                    op_index].input(input_attr_name)[0]
                                if self.__check_var_source_dt(
                                        input_var_name) != -1:
                                    op_has_int8_input = True
                                    break

                        if op_has_int8_input:
                            if op_type == "conv2d":
                                if program.current_block().ops[
                                        op_index + 1].type == "conv2d":
                                    continue
                                elif program.current_block(
                                ).ops[op_index +
                                      1].type in Calibrator.non_conv_int8_op_type:
                                    dequantize_op_index.append(op_index + 1)
                                    break
                                else:
                                    program.current_block().ops[
                                        op_index]._set_attr("force_fp32_output",
                                                            True)
                                    continue
                            else:
                                if self.__check_force_fp32_attr_by_output_var(
                                        program, input_var_name):
                                    pass
                                else:
                                    if op_index not in dequantize_op_index:
                                        share_input_flag = True
                                        for input_attr_name in program.current_block(
                                        ).ops[op_index].input_names:
                                            input_var_name = program.current_block(
                                            ).ops[op_index].input(
                                                input_attr_name)[0]
                                            cousin_op_index = self.__get_op_index_by_input_var(
                                                program, input_var_name)
                                            if cousin_op_index != -1 and cousin_op_index in dequantize_op_index:
                                                share_input_flag = False
                                                break
                                        if share_input_flag:
                                            dequantize_op_index.append(op_index)
                        else:
                            if input_length:
                                output_is_to_int8_op = False
                                share_input_flag = True
                                for input_attr_name in program.current_block(
                                ).ops[op_index].input_names:
                                    input_var_name = program.current_block(
                                    ).ops[op_index].input(input_attr_name)[0]
                                    if not self.__check_op_type_with_specified_var_as_input(
                                            program, input_var_name):
                                        share_input_flag = False
                                        break

                                for output_attr_name in program.current_block(
                                ).ops[op_index].output_names:
                                    output_var_name = program.current_block(
                                    ).ops[op_index].output(output_attr_name)[0]
                                    if self.__get_op_index_by_output_var(
                                            program, output_var_name,
                                            op_index) != -1:
                                        output_is_to_int8_op = True
                                        break

                                if share_input_flag or output_is_to_int8_op:
                                    quantize_op_index.append(op_index - 1)
                else:
                    continue

        return quantize_op_index, dequantize_op_index

    def __init_analysis(self):
        for i in self._conv_op_index[1:]:
            self._weights_var_name.append(self.program.current_block().ops[i]
                                          .input('Filter')[0])
            self._conv_input_var_name.append(self.program.current_block().ops[i]
                                             .input('Input')[0])
            self._conv_output_var_name.append(self.program.current_block().ops[
                i].output('Output')[0])
            self._int8_output_var_op_index_dict[self.program.current_block()
                                                .ops[i].output('Output')[0]] = i
            if self.program.current_block().ops[i].desc.input("ResidualData"):
                self._residual_input_var_name.append(self.program.current_block(
                ).ops[i].desc.input("ResidualData")[0])

            if self.program.current_block().ops[i + 1].type == "pool2d":
                self._conv_output_var_name.append(self.program.current_block()
                                                  .ops[i + 1].output('Out')[0])

    def __expand_quantized_bins(self, quantized_bins, reference_bins):
        expanded_quantized_bins = [0] * len(reference_bins)
        num_merged_bins = len(reference_bins) / len(quantized_bins)
        j_start = 0
        j_end = num_merged_bins
        for idx in xrange(len(quantized_bins)):
            zero_count = reference_bins[j_start:j_end].count(0)
            num_merged_bins = j_end - j_start
            if zero_count == num_merged_bins:
                avg_bin_ele = 0
            else:
                avg_bin_ele = quantized_bins[idx] / (
                    num_merged_bins - zero_count + 0.0)
            for idx1 in xrange(j_start, j_end):
                expanded_quantized_bins[idx1] = (0 if reference_bins[idx1] == 0
                                                 else avg_bin_ele)
            j_start += num_merged_bins
            j_end += num_merged_bins
            if (idx + 1) == len(quantized_bins) - 1:
                j_end = len(reference_bins)
        return expanded_quantized_bins

    def __safe_entropy(self, reference_distr_P, P_sum, candidate_distr_Q,
                       Q_sum):
        assert len(reference_distr_P) == len(candidate_distr_Q)
        tmp_sum1 = 0
        tmp_sum2 = 0
        for idx in range(len(reference_distr_P)):
            p_idx = reference_distr_P[idx]
            q_idx = candidate_distr_Q[idx]
            if p_idx == 0:
                tmp_sum1 += 0
                tmp_sum2 += 0
            else:
                if q_idx == 0:
                    print "Fatal error!, idx = " + str(
                        idx) + " qindex = 0! p_idx = " + str(p_idx)
                tmp_sum1 += p_idx * (math.log(Q_sum * p_idx))
                tmp_sum2 += p_idx * (math.log(P_sum * q_idx))
        return (tmp_sum1 - tmp_sum2) / P_sum

    def __get_optimal_scaling_factor(self,
                                     activation_blob,
                                     num_quantized_bins=255):
        max_val = np.max(activation_blob)
        min_val = np.min(activation_blob)
        if min_val >= 0:
            hist, hist_edeges = np.histogram(
                activation_blob, bins=2048, range=(min_val, max_val))
            ending_iter = 2047
            starting_iter = int(ending_iter * 0.7)
        else:
            th = max(abs(max_val), abs(min_val))
            hist, hist_edeges = np.histogram(
                activation_blob, bins=2048, range=(-th, th))
            starting_iter = 0
            ending_iter = 2047
            if abs(max_val) > abs(min_val):
                while starting_iter < ending_iter:
                    if hist[starting_iter] == 0:
                        starting_iter += 1
                        continue
                    else:
                        break
                starting_iter += int((ending_iter - starting_iter) * 0.6)
            else:
                while ending_iter > 0:
                    if hist[ending_iter] == 0:
                        ending_iter -= 1
                        continue
                    else:
                        break
                starting_iter = int(0.6 * ending_iter)
        bin_width = hist_edeges[1] - hist_edeges[0]
        P_sum = len(activation_blob)
        min_kl_divergence = 0
        min_kl_index = 0
        kl_inited = False
        for i in range(starting_iter, ending_iter + 1):
            reference_distr_P = hist[0:i].tolist()
            outliers_count = sum(hist[i:2048])
            if reference_distr_P[i - 1] == 0:
                continue
            reference_distr_P[i - 1] += outliers_count
            reference_distr_bins = reference_distr_P[:]
            candidate_distr_Q = hist[0:i].tolist()
            num_merged_bins = i / num_quantized_bins
            candidate_distr_Q_quantized = [0] * num_quantized_bins
            j_start = 0
            j_end = num_merged_bins
            for idx in xrange(num_quantized_bins):
                candidate_distr_Q_quantized[idx] = sum(candidate_distr_Q[
                    j_start:j_end])
                j_start += num_merged_bins
                j_end += num_merged_bins
                if (idx + 1) == num_quantized_bins - 1:
                    j_end = i
            candidate_distr_Q = self.__expand_quantized_bins(
                candidate_distr_Q_quantized, reference_distr_bins)
            Q_sum = sum(candidate_distr_Q)
            kl_divergence = self.__safe_entropy(reference_distr_P, P_sum,
                                                candidate_distr_Q, Q_sum)
            if not kl_inited:
                min_kl_divergence = kl_divergence
                min_kl_index = i
                kl_inited = True
            elif kl_divergence < min_kl_divergence:
                min_kl_divergence = kl_divergence
                min_kl_index = i
            else:
                pass
        if min_kl_index == 0:
            while starting_iter > 0:
                if hist[starting_iter] == 0:
                    starting_iter -= 1
                    continue
                else:
                    break
            min_kl_index = starting_iter
        return (min_kl_index + 0.5) * bin_width

    @staticmethod
    def __dot(program, output_name="model.dot"):
        '''
        '''
        dot_graph = ""
        dot_nodes = []
        dot_edges = []
        dot_graph += "digraph pm {\n"
        for block in program.blocks:
            ops = list(block.ops)
            for index, op in enumerate(ops):
                op_type = op.type
                op_name = op_type + "_" + op.output_arg_names[0].replace(
                    ".", "_") + "___" + str(index)
                for name in op.input_arg_names:
                    name = name.replace(".", "_")
                    dot_edge = name + " -> " + op_name
                    if dot_edge not in dot_edges:
                        dot_edges.append(dot_edge)
                    dot_node = name + " [shape=oval, style=filled, fillcolor=yellow]"
                    if dot_node not in dot_nodes:
                        dot_nodes.append(dot_node)

                for name in op.output_arg_names:
                    name = name.replace(".", "_")
                    dot_edge = op_name + " -> " + name
                    if dot_edge not in dot_edges:
                        dot_edges.append(dot_edge)
                if op_type in Calibrator.supported_int8_op_type:
                    if op_type == "conv2d" and op.has_attr(
                            'force_fp32_output') and op.attr(
                                "force_fp32_output"):
                        dot_node = op_name + " [shape=box, style=filled, color=deeppink]"
                    else:
                        dot_node = op_name + " [shape=box, style=filled, color=greenyellow]"
                elif op_type in ["quantize", "dequantize"]:
                    dot_node = op_name + " [shape=box, style=filled, color=gold]"
                else:
                    dot_node = op_name + " [shape=box, style=filled, fillcolor=red]"

                if dot_node not in dot_nodes:
                    dot_nodes.append(dot_node)

        for dot_edge in dot_edges:
            dot_graph += dot_edge + "\n"
        for dot_node in dot_nodes:
            dot_graph += dot_node + "\n"
        dot_graph += "}"

        with open(output_name, 'w') as f:
            f.write(dot_graph)
