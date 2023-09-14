# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import re

# this is a file's header part
CommonHead = '''
// Generated by generic_mixed_gemm_kernelLauncher.py - Do not edit.

#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace phi {
'''

CommonTail = '''
} // namespace phi

'''
DispatchGemmConfigInstanceDeclare = """
template<>
void generic_mixed_gemm_kernelLauncher_template<{T},
                                                {WeightType},
                                                {arch},
                                                {EpilogueTag},
                                                {ThreadblockShape},
                                                {WarpShape},
                                                {Stages}>(
    const {T}* A,
    const {WeightType}* B,
    const float* weight_scales,
    const {T}* biases,
    {T}* C,
    int m,
    int n,
    int k,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy) {
    generic_mixed_gemm_kernelLauncher<{T},
                                      {WeightType},
                                      {arch},
                                      {EpilogueTag},
                                      {ThreadblockShape},
                                      {WarpShape},
                                      {Stages}>(
        A,
        B,
        weight_scales,
        biases,
        C,
        m,
        n,
        k,
        gemm_config,
        workspace,
        workspace_bytes,
        stream,
        occupancy);
}
"""

DefineHeader = """
// Generated by generic_mixed_gemm_kernelLauncher.py - Do not edit.

"""

DefaultArch = [70, 75, 80]
epilogue_tags = ["bias", "biasFtGelu", "biasReLU", "noBias"]

WeightTypes = ["uint8_t", "cutlass::uint4b_t"]
ThreadblockShapes = [
    "cutlass::gemm::GemmShape<32, 128, 64>",
    "cutlass::gemm::GemmShape<64, 128, 64>",
    "cutlass::gemm::GemmShape<128, 128, 64>",
    "cutlass::gemm::GemmShape<256, 128, 64>",
    "cutlass::gemm::GemmShape<128, 256, 64>",
]
WarpShapes = [
    "cutlass::gemm::GemmShape<32, 32, 64>",
    "cutlass::gemm::GemmShape<64, 32, 64>",
    "cutlass::gemm::GemmShape<128, 32, 64>",
    "cutlass::gemm::GemmShape<64, 64, 64>",
    "cutlass::gemm::GemmShape<64, 64, 64>",
]
StagesList = {70: [2], 75: [2], 80: [2, 3, 4]}

ElementTypes = {"fp16": "half", "bf16": "__nv_bfloat16"}
Archs = {
    70: "cutlass::arch::Sm70",
    75: "cutlass::arch::Sm75",
    80: "cutlass::arch::Sm80",
}
EpilogueTags = {
    "bias": "EpilogueOpBias",
    "biasFtGelu": "EpilogueOpBiasFtGelu",
    "biasReLU": "EpilogueOpBiasReLU",
    "noBias": "EpilogueOpNoBias",
}


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def find_arch_range(archs):
    compile_archs = []
    for arch in archs:
        if arch >= 70 and arch < 75:
            compile_archs.append(70)
        elif arch >= 75 and arch < 80:
            compile_archs.append(75)
        elif arch >= 80 and arch < 90:
            compile_archs.append(80)
    compile_archs = list(set(compile_archs))
    compile_archs.sort()
    return compile_archs


def convert_to_arch_list(archs):
    archs = archs.lower().strip()
    if archs == "all":
        return DefaultArch

    archs = [int(s.strip()) for s in archs.split(';') if s.strip()]
    archs = list(set(archs))
    return find_arch_range(archs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="The argument for generating the generic_mixed_gemm_kernelLauncher instance."
    )
    parser.add_argument(
        "--cuda_arch",
        type=convert_to_arch_list,
        default=convert_to_arch_list("All"),
        help="The CUDA architecture to be generated.",
    )
    args = parser.parse_args()
    return args


# generate source cu
def generate_source_cu(
    element_type: str, arch: int, epilogue_tag: str, stages: int
):
    all_code = CommonHead
    for WeightType in WeightTypes:
        for i in range(len(ThreadblockShapes)):
            value_dict = {
                "T": ElementTypes[element_type],
                "WeightType": WeightType,
                "arch": Archs[arch],
                "EpilogueTag": EpilogueTags[epilogue_tag],
                "ThreadblockShape": ThreadblockShapes[i],
                "WarpShape": WarpShapes[i],
                "Stages": str(stages),
            }
            all_code += SubstituteTemplate(
                DispatchGemmConfigInstanceDeclare, value_dict
            )
    all_code += CommonTail
    return all_code


if __name__ == "__main__":
    args = parse_args()
    archs = args.cuda_arch
    header_all = DefineHeader
    header_name = "autogen/arch_define.h"
    if archs:
        for arch in archs:
            define_line = "#define USE_FPAINTB_GEMM_WITH_SM%s\n" % str(arch)
            header_all += define_line
    with open(header_name, "w") as f:
        f.write(header_all)
        f.close()
    if archs:
        for element_type in ElementTypes.keys():
            for arch in archs:
                for epilogue_tag in EpilogueTags.keys():
                    for stages in StagesList[arch]:
                        file_name = "autogen/generic_mixed_gemm_kernelLauncher_{}_sm{}_stages{}_{}.cu".format(
                            element_type, arch, stages, epilogue_tag
                        )
                        all_code = generate_source_cu(
                            element_type, arch, epilogue_tag, stages
                        )
                        with open(file_name, "w") as f:
                            f.write(all_code)
                            f.close()
