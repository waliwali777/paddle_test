import unittest

import numpy
import paddle.v2.framework.core as core
import paddle.v2.framework.create_op_creation_methods as creation
from paddle.v2.framework.create_op_creation_methods import op_creations

__all__ = ['get_numeric_gradient']


def create_op(op_type):
    func = getattr(creation.op_creations, op_type, None)
    assert (func is not None)

    kwargs = dict()
    for in_name in func.all_input_args:
        kwargs[in_name] = in_name
    for out_name in func.all_output_args:
        kwargs[out_name] = out_name
    op = func(**kwargs)
    return op


def grad_var_name(var_name):
    return var_name + "@GRAD"


def get_numeric_gradient(op,
                         input_values,
                         output_name,
                         input_to_check,
                         delta=0.005,
                         local_scope=None):
    """
    Get Numeric Gradient for an operator's input.
    
    :param op: C++ operator instance, could be an network 
    :param input_values: The input variables. Should be an dictionary, key is 
    variable name. Value is numpy array.
    :param output_name: The final output variable name. 
    :param input_to_check: The input variable need to get gradient.
    :param delta: The perturbation value for numeric gradient method. The 
    smaller delta is, the more accurate result will get. But if that delta is
     too small, it could occur numerical stability problem.
    :param local_scope: The local scope used for get_numeric_gradient.
    :return: The gradient array in numpy format.
    """
    if local_scope is None:
        local_scope = core.Scope()

    # Create all input variable in local_scope
    for var_name in input_values:
        var = local_scope.new_var(var_name)
        tensor = var.get_tensor()
        tensor.set_dims(input_values[var_name].shape)
        tensor.alloc_float(core.CPUPlace())
        tensor.set(input_values[var_name], core.CPUPlace())

    # Create all output variable in local_scope
    for output in op.outputs():
        if local_scope.find_var(output) is None:
            local_scope.new_var(output).get_tensor()

    op.infer_shape(local_scope)

    # allocate output memory
    for output in op.outputs():
        local_scope.find_var(output).get_tensor().alloc_float(core.CPUPlace())

    # TODO(yuyang18): Only CPU is support now.
    cpu_ctx = core.DeviceContext.create(core.CPUPlace())

    def get_output():
        op.run(local_scope, cpu_ctx)
        return numpy.array(local_scope.find_var(output_name).get_tensor()).sum()

    def product(dim):
        return reduce(lambda a, b: a * b, dim, 1)

    tensor_to_check = local_scope.find_var(input_to_check).get_tensor()
    tensor_size = product(tensor_to_check.get_dims())
    gradient_flat = numpy.zeros(shape=(tensor_size, ), dtype='float32')
    for i in xrange(tensor_size):
        origin = tensor_to_check.get_float_element(i)
        x_pos = origin + delta
        tensor_to_check.set_float_element(i, x_pos)
        y_pos = get_output()

        x_neg = origin - delta
        tensor_to_check.set_float_element(i, x_neg)
        y_neg = get_output()

        tensor_to_check.set_float_element(i, origin)  # restore old value
        gradient_flat[i] = (y_pos - y_neg) / delta / 2
    return gradient_flat.reshape(tensor_to_check.get_dims())


class GradientChecker(unittest.TestCase):
    def assert_grad(self,
                    forward_op,
                    input_vars,
                    inputs_to_check,
                    output_name,
                    no_grad_set=set(),
                    only_cpu=False):
        out_names = filter(lambda name: name != "@TEMP@", forward_op.outputs())
        if len(out_names) != 1:
            raise ValueError("non empty out_names should be 1")

        in_names = forward_op.inputs()
        for no_grad in no_grad_set:
            if no_grad not in in_names:
                raise ValueError("no_grad should be in in_names")

        for check_name in inputs_to_check:
            if check_name not in in_names:
                raise ValueError("check name should be in in_names")

        backward_op = core.Operator.backward(forward_op, no_grad_set)

        places = [core.CPUPlace()]
        if core.is_compile_gpu() and not only_cpu:
            places.append(core.GPUPlace(0))

        numeric_grad = dict()
        cpu_grad = dict()
        gpu_grad = dict()

        for place in places:
            scope = core.Scope()
            ctx = core.DeviceContext.create(place)

            # create input var and set value
            for name, value in input_vars.iteritems():
                assert name in in_names
                var = scope.new_var(name).get_tensor()
                var.set_dims(value.shape)
                var.set(value, place)

            # create output var
            for out_name in forward_op.outputs():
                scope.new_var(out_name).get_tensor()

            # infer the shape of output var and set value of output var
            forward_op.infer_shape(scope)
            forward_op.run(scope, ctx)

            # create output grad var
            # set shape as the output var
            # set value of this grad to ones
            for name in forward_op.outputs():
                out_tensor = scope.find_var(name).get_tensor()
                grad_tensor = scope.new_var(grad_var_name(name)).get_tensor()
                grad_tensor.set_dims(out_tensor.shape())
                data = 1.0 * numpy.ones(out_tensor.shape())
                grad_tensor.set(data, place)

            # create input grad var
            for name in backward_op.outputs():
                scope.new_var(name).get_tensor()

            backward_op.infer_shape(scope)
            backward_op.run(scope, ctx)

            numeric_input = dict()
            for name in forward_op.inputs():
                data = numpy.array(scope.find_var(name).get_tensor())
                numeric_input[name] = data
            for check_name in inputs_to_check:
                # get and store numeric_grad
                if not numeric_grad.has_key(check_name):
                    numeric_grad[check_name] = \
                        get_numeric_gradient(forward_op, numeric_input, output_name, check_name)
                # get xpu_grad
                op_grad = numpy.array(
                    scope.find_var(grad_var_name(check_name)).get_tensor())

                # check numeric_grad early with CPUPlace
                if isinstance(place, core.CPUPlace):
                    self.assertTrue(
                        numpy.allclose(
                            numeric_grad[check_name],
                            op_grad,
                            rtol=0.05,
                            atol=100),
                        "numeric_gradient and cpu op gradient are not equal")

                # store cpu grad
                if isinstance(place, core.CPUPlace):
                    cpu_grad[check_name] = op_grad
                # store gpu grad
                if isinstance(place, core.GPUPlace):
                    gpu_grad[check_name] = op_grad

        # compile cpu and gpu grad
        if core.is_compile_gpu() and not only_cpu:
            for check_name in cpu_grad:
                self.assertTrue(
                    numpy.allclose(
                        cpu_grad[check_name],
                        gpu_grad[check_name],
                        rtol=0.05,
                        atol=100),
                    "cpu op gradient and gpu op gradient are not equal")


if __name__ == '__main__':

    class GetNumericGradientTest(unittest.TestCase):
        def test_add_op(self):
            add_op = op_creations.add_two(X="X", Y="Y", Out="Z")
            x = numpy.random.random((10, 1)).astype("float32")
            y = numpy.random.random((10, 1)).astype("float32")

            arr = get_numeric_gradient(add_op, {'X': x, "Y": y}, 'Z', 'X')
            self.assertAlmostEqual(arr.mean(), 1.0, delta=1e-2)

        def test_softmax_op(self):
            def stable_softmax(x):
                """Compute the softmax of vector x in a numerically stable way."""
                shiftx = x - numpy.max(x)
                exps = numpy.exp(shiftx)
                return exps / numpy.sum(exps)

            def label_softmax_grad(Y, dY):
                dX = Y * 0.0
                for i in range(Y.shape[0]):
                    d = numpy.dot(Y[i, :], dY[i, :])
                    dX[i, :] = Y[i, :] * (dY[i, :] - d)
                return dX

            softmax_op = op_creations.softmax(X="X", Y="Y")

            X = numpy.random.random((2, 2)).astype("float32")
            Y = numpy.apply_along_axis(stable_softmax, 1, X)
            dY = numpy.ones(Y.shape)
            dX = label_softmax_grad(Y, dY)

            arr = get_numeric_gradient(softmax_op, {"X": X}, 'Y', 'X')
            numpy.testing.assert_almost_equal(arr, dX, decimal=1e-2)

    unittest.main()
