import os
import tempfile
import copy
import warnings
import inspect
import re

import torch
from torch._six import PY2
import torch.testing._internal.common_utils as common
import torch.testing._internal.common_nn as common_nn
import torch.utils.cpp_extension
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity import sample_module, torch_nn_modules
from cpp_api_parity import functional_impl_check, module_impl_check

class TestCppApiParity(common.TestCase):
  pass

parity_table_path = os.path.join(os.path.dirname(__file__), 'cpp_api_parity/parity-tracker.md')

parity_table = parse_parity_tracker_table(parity_table_path)

# module_tests = common_nn.module_tests + common_nn.new_module_tests + sample_module.module_tests
module_tests = []
for padding_mode in ['reflect', 'circular', 'replicate', 'zeros']:
# for padding_mode, cpp_padding_mode in zip(
#     ['reflect', 'circular', 'replicate', 'zeros'],
#     ['torch::kReflect', 'torch::kCircular', 'torch::kReplicate', 'torch::kZeros']):
    cpp_padding_mode = 'torch::kReflect'
    # conv signature:
    #     in_channels, out_channels, kernel_size, stride=1,
    #     padding=0, dilation=1, groups=1,
    #     bias=True, padding_mode='zeros'
    for d in (1, 2, 3):
        if d == 3 and padding_mode == 'reflect':
            # FIXME: remove after implementing reflection pad 3d
            #        https://github.com/pytorch/pytorch/issues/27655
            continue
        input_size = (2, 3) + (3,) * d
        cpp_input_args = ['torch::randn({{}})'.format(', '.join([str(x) for x in input_size]))]
        module_tests.append(
            dict(
                module_name='Conv{}d'.format(d),
                constructor_args=(3, 4, 3, 2, 2, 1, 1, True, "circular"),
                cpp_constructor_args='torch::nn::Conv'+str(d)+'dOptions(3, 4, 3).stride(2).padding(2).dilation(1).groups(1).bias(true).padding_mode('+str(cpp_padding_mode)+')',
                input_size=(2, 3) + (3,) * d,
                cpp_input_args=cpp_input_args,
                output_size=(2, 4) + (3,) * d,
                cudnn=True,
                desc='{}_stride2_pad2'.format(padding_mode),
            ),
        )
# criterion_tests = common_nn.criterion_tests + common_nn.new_criterion_tests

# module_tests = [
#     dict(
#         module_name='Hardtanh',
#         input_size=(3, 2, 5),
#         cpp_input_args=['torch::randn({3, 2, 5})'],
#         reference_fn=lambda i, *_: i.clamp(-1, 1),
#     )
# ]
criterion_tests = [] # yf225 TODO: change back to actual lists

module_impl_check.add_tests(TestCppApiParity, module_tests, criterion_tests, torch_nn_modules, parity_table)
# functional_impl_check.add_tests(module_tests, criterion_tests, torch_nn_modules, parity_table)

if __name__ == "__main__":
  common.run_tests()
