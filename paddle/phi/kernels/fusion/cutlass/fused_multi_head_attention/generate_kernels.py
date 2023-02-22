# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Generates combination of kernels - implementations and registry

# Kernels are ordered (see `sort_index`), and when dispatching,
# we select the first kernel in the list that supports the inputs

import collections
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar

# TODO(zhengzekang): Currently we only register FP16 kernel. 
DTYPES = {
    # "f32": "float",
    "f16": "cutlass::half_t",
}

SM = [70, 75, 80]

KERNEL_IMPL_TEMPLATE = """
template<>
__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}<{CPP_CLASS}>(typename {CPP_CLASS}::Params params) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= {SM}0 && __CUDA_ARCH__ < {SM_MAX}0
  if (!params.advance_to_block()) {{
    return;
  }}
  {CPP_CLASS}::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `{NAME}` is for sm{SM}-sm{SM_MAX}, but was built for sm%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}}
"""


@dataclass(order=True)
class FwdKernel:
    sort_index: Tuple[int, ...] = field(init=False, repr=False)
    aligned: bool
    dtype: str
    sm_range: Tuple[int, int]
    q: int
    k: int
    single_value_iter: bool
    add_mask: bool = True
    mask_broadcast_row: bool = True
    dispatch_cond: Optional[str] = None

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Then keep output in RF
            0 if self.single_value_iter else 1,
            self.k,
            # Prefer kernels without dropout/bias if available
            1 if self.add_mask else 0,
            1 if self.mask_broadcast_row else 0,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def _sm_suffix(self) -> str:
        return self.sm_range[0]

    @property
    def name(self) -> str:
        # acc = "rf" if self.single_value_iter else "gmem"
        # return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_{self.q}x{self.k}_{acc}_sm{self.sm_range[0]}"
        return f"attention_kernel_batched"

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.q),
                str(self.k),
                "true" if self.single_value_iter else "false",
                "true" if self.add_mask else "false",
                "true" if self.mask_broadcast_row else "false",
            ]
        )
        return f"AttentionKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        # For example: cutlass_fmha_forward_f16_aligned_70.cu contains kernel with fp16, Aligned, SM70 implementation. 
        return f"{self.dtype}_{self._aligned_suffix}_{self._sm_suffix}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> List["FwdKernel"]:
        kernels: List[FwdKernel] = []
        for aligned, dtype, (sm, sm_max) in itertools.product(
            [True], DTYPES.keys(), zip(SM, SM[1:] + [90])
        ):
            # Remove some kernels we don't use
            if dtype == "bf16" and sm < 80:
                continue
            for q, k, single_value_iter in [
                (32, 128, True),
                (32, 128, False),
                (64, 64, True),
            ]:
                for add_mask, mask_broadcast_row in [
                    (True, True), 
                    (True, False), 
                    (False, True), 
                    (False, False), 
                ]: 
                    kernels.append(
                        cls(
                            aligned=aligned,
                            dtype=dtype,
                            sm_range=(sm, sm_max),
                            q=q,
                            k=k,
                            single_value_iter=single_value_iter,
                            add_mask=add_mask, 
                            mask_broadcast_row=mask_broadcast_row
                        )
                    )
        return kernels


# T = TypeVar("T", FwdKernel, BwdKernel)
T = FwdKernel

def write_decl_impl(
    kernels: List[T], family_name: str, impl_file: str, disable_def: str
) -> None:
    cpp_file_header = """// This file is auto-generated. See "generate_kernels.py"
"""

    kernels.sort()

    implfile_to_kernels: Dict[str, List[T]] = collections.defaultdict(list)
    cat_to_kernels: Dict[Tuple[str, int, int], List[T]] = collections.defaultdict(list)

    dispatch_all = ""
    declarations = cpp_file_header + "#pragma once\n"
    declarations += f"#ifndef {disable_def}\n"
    declarations += f"""#include "../{impl_file}"\n"""

    # Declaration of kernel functions
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        cat_to_kernels[(k.dtype, k.sm_range[0], k.sm_range[1])].append(k)

    for (cat_dt, cat_sm, cat_sm_max), kernels in cat_to_kernels.items():
        declarations += f"\n// ======== {cat_dt} / sm{cat_sm} ========\n"
        declarations += "\n".join(
            k.cpp_impl.split("{")[0].rstrip() + ";" for k in kernels
        )
        declarations += "\n\n"

    declarations += f"#endif // {disable_def}\n"

    autogen_dir = Path(__file__).parent / "kernels"

    (autogen_dir / f"{family_name}.h").write_text(declarations)

    for f, f_kernels in implfile_to_kernels.items():
        impl_cu = cpp_file_header
        impl_cu += f"#ifndef {disable_def}\n"
        impl_cu += f"""#include "../../{impl_file}"\n"""
        for k in f_kernels:
            impl_cu += k.cpp_impl
        impl_cu += f"#endif // {disable_def}\n"
        (autogen_dir / "impl" / f"{family_name}_{f}.cu").write_text(impl_cu)


write_decl_impl(
    FwdKernel.get_all(),
    "cutlass_fmha_forward",
    impl_file="kernel_forward.h",
    disable_def="XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD",
)
