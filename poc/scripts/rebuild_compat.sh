#!/bin/bash
# =============================================================================
# 重建 sgl-kernel 兼容层 (每次重启后运行一次)
# =============================================================================
# 解决的问题:
#   - sgl-kernel 0.3.21 的 CUDA 内核只为 SM90/SM100 编译
#   - RTX 3090 是 SM86, 需要 patch 让 sgl-kernel 加载 SM90 内核
#   - PyTorch 2.13 与 sgl-kernel 0.3.21 有 C++ ABI 不匹配 (6 处符号)
#   - 通过 ASM shim 桥接这些符号差异
# =============================================================================

set -e

TORCH_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/torch/lib"
SGKERNEL="/home/wangwei/.local/lib/python3.13/site-packages/sgl_kernel"

echo "[1/3] 二进制 patch: sgl-kernel common_ops (c10_cuda_check 符号)"
/usr/bin/python3.13 -c "
path = '${SGKERNEL}/sm90/common_ops.abi3.so'
with open(path, 'rb') as f: data = f.read()
old = b'_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib'
new = b'_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb'
if old in data:
    data = data.replace(old, new)
    with open(path, 'wb') as f: f.write(data)
    print('  patched')
else:
    print('  already patched')
"

echo "[2/3] Patch load_utils.py (SM86 → SM90 路由)"
if ! grep -q "compute_capability >= 80" ${SGKERNEL}/load_utils.py; then
    /usr/bin/python3.13 -c "
path = '${SGKERNEL}/load_utils.py'
with open(path) as f: content = f.read()
content = content.replace(
    'elif compute_capability is not None:',
    'elif compute_capability is not None and compute_capability >= 80:\n        ops_subdir = \"sm90\"  # SM80-89 use sm90 kernels\n        variant_name = f\"SM{compute_capability} (SM90 compat)\"'
)
with open(path, 'w') as f: f.write(content)
print('  patched')
"
else
    echo "  already patched"
fi

echo "[3/3] 构建 ASM 兼容 shim (/tmp/libsgl_compat_all.so)"
/usr/bin/python3.13 << 'PYEOF'
asm = r""".text
.globl _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v
_ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v:
    jmp _ZNK2at10TensorBase14const_data_ptrIiEEPKT_v
.globl _ZN2at4cuda24getCurrentCUDABlasHandleEv
_ZN2at4cuda24getCurrentCUDABlasHandleEv:
    xor %esi,%esi
    jmp _ZN2at4cuda24getCurrentCUDABlasHandleEb
.globl _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
_ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE:
    jmp _ZN3c104impl3cow15materialize_cowEPNS_11StorageImplE
.globl _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE
_ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE:
    jmp _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorINS_10headeronly3TagESaIS8_EENS_17_RegisterOrVerifyE
.globl _ZNR5torch7Library4_defEOSt7variantIJN3c1012OperatorNameENS2_14FunctionSchemaEEEONS_11CppFunctionERKSt6vectorIN2at3TagESaISB_EE
_ZNR5torch7Library4_defEOSt7variantIJN3c1012OperatorNameENS2_14FunctionSchemaEEEONS_11CppFunctionERKSt6vectorIN2at3TagESaISB_EE:
    jmp _ZNR5torch7Library4_defEOSt7variantIJN3c1012OperatorNameENS2_14FunctionSchemaEEEONS_11CppFunctionERKSt6vectorINS_10headeronly3TagESaISB_EE
.globl _ZN3c1013MessageLoggerC1EPKcii
_ZN3c1013MessageLoggerC1EPKcii:
    push %rbp; mov %rsp,%rbp; push %rbx; mov %rdi,%rbx; mov %esi,%r8d
    sub $32,%rsp; mov %rsi,(%rsp); movq $0,8(%rsp); movq $0,16(%rsp)
    mov %rbx,%rdi; lea (%rsp),%rsi; mov %r8d,%edx; xor %ecx,%ecx
    call _ZN3c1013MessageLoggerC1ENS_14SourceLocationEib
    add $32,%rsp; pop %rbx; pop %rbp; ret
.globl _ZN3c1013MessageLoggerC2EPKcii
_ZN3c1013MessageLoggerC2EPKcii:
    jmp _ZN3c1013MessageLoggerC1EPKcii
"""
with open('/tmp/shim.s', 'w') as f:
    f.write(asm)
PYEOF

as -o /tmp/shim.o /tmp/shim.s
gcc -shared -o /tmp/libsgl_compat_all.so /tmp/shim.o \
    -L${TORCH_LIBS} -Wl,-rpath,${TORCH_LIBS} -Wl,--no-as-needed \
    ${TORCH_LIBS}/libc10.so ${TORCH_LIBS}/libtorch_cpu.so -lstdc++

echo ""
echo "✅ 兼容层构建完成"
echo "   使用: export LD_PRELOAD=/tmp/libsgl_compat_all.so"
