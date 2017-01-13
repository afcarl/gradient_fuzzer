"""Fuzz single argument ops' gradients in tensorflow."""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import inspect
import itertools

IGNORE_ERRORS = (LookupError, TypeError, AssertionError, NotImplementedError,
              ValueError, AttributeError, tf.errors.InvalidArgumentError)


def get_all_methods(library):
  """Return all methods of a library."""
  members = inspect.getmembers(library)
  return [m for m in members if inspect.isfunction(m[1])]


def has_single_arg(op):
  """Check whether a tensorflow op takes a single argument."""
  args = inspect.getargspec(op[1])[0]
  if len(args) == 2 and 'name' in args and 'x' in args:
    return True
  else:
    return False


def build_single_arg_op(fn, op_tuple, x):
  """Build op for a lambda function."""
  op_name, op = op_tuple
  try:
    y = fn(op, x)
    name = 'y = %s(%s(x))' % (fn.__name__, op_name)
    return name, y
  except IGNORE_ERRORS:
    return


def build_chain(fn, two_ops, x):
  """Build chain of two ops applied to x."""
  first_op_tuple, second_op_tuple = two_ops
  first_op_name, first_op = first_op_tuple
  second_op_name, second_op = second_op_tuple
  try:
    y = fn(first_op, second_op, x)
    name = 'y = %s(%s(%s(x))' % (fn.__name__, first_op_name, second_op_name)
    return name, y
  except IGNORE_ERRORS:
    return


def _identity(op, x):
  """Op applied to x."""
  return op(x)


def _gradient(op, x):
  """Gradient of op(x) with respect to x."""
  return tf.gradients(op(x), x)[0]


def _chain(first_op, second_op, x):
  """chain of two functions."""
  return first_op(second_op(x))


def _gradient_chain(first_op, second_op, x):
  """Gradient of chain of two functions."""
  return tf.gradients(first_op(second_op(x)), x)[0]


def fuzz_node_list(node_list, x, num_points=10, tol=1e-1):
  """Test a gradients of a node on a log scale."""
  for node_tuple in node_list:
    name, y = node_tuple
    print 'Testing gradient of %s' % name
    with tf.Session():
      for point in np.logspace(-100, 100, num=num_points):
        try:
          error = tf.test.compute_gradient_error(
              x, [], y, [], extra_feed_dict={x: point})
          if error > tol:
            print ('For %s' % name +
                   ' point %.3e, ' % point +
                   'gradient error was %.3e ' % error +
                   'at tolerance %.3e ' % tol)
        except IGNORE_ERRORS:
          pass


def main(_):
  single_arg_ops = filter(has_single_arg, get_all_methods(math_ops))
  # Test single argument ops
  x = tf.placeholder(tf.float32, [])
  remove_nones = lambda lst: [e for e in lst if e is not None]
  # Test single ops
  y_list = [build_single_arg_op(_identity, op, x) for op in single_arg_ops]
  y_list = remove_nones(y_list)
  fuzz_node_list(y_list, x)
  # Test gradients of gradients of single ops
  y_list = [build_single_arg_op(_gradient, op, x) for op in single_arg_ops]
  y_list = remove_nones(y_list)
  fuzz_node_list(y_list, x)
  # Test chain of two ops
  perms = list(itertools.permutations(single_arg_ops, 2))
  y_list = [build_chain(_chain, two_ops, x) for two_ops in perms]
  y_list = remove_nones(y_list)
  fuzz_node_list(y_list, x)
  # Test gradient of chain of two ops
  perms = list(itertools.permutations(single_arg_ops, 2))
  y_list = [build_chain(_gradient_chain, two_ops, x) for two_ops in perms]
  y_list = remove_nones(y_list)
  fuzz_node_list(y_list, x)



if __name__ == '__main__':
  tf.app.run()
