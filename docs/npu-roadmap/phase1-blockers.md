# Phase 1 hwctx blocker — RESOLVED 2026-05-01

## Original problem (recap)

CREATE_HWCTX reached the firmware's `MSG_OP_MAP_HOST_BUFFER` and was
rejected with status `0x4000003` (BAD_PARAMETER) on every parameter
combination. The Ubuntu in-tree `amdxdna` driver on Linux 7.0 does not
implement hugepage-backed BO mmap that the AIE-2P (`npu_7`) firmware
requires for the IOMMU-mapped heap. xrt-smi from the
`libxrt-utils-npu` package hit the *same* wall.

## Resolution

Built and installed the out-of-tree `amd/xdna-driver` from
github.com/amd/xdna-driver via DKMS:

- Module size went from 172 KB (in-tree v0.7.0) → **606 KB (OOT v1.0.0)**
  — 3.5× more code, the missing capabilities are real.
- dmesg on init now reports **`PASID address mode enabled`** — SVM-style
  IOMMU with per-process address spaces, the right model for our
  CPU+iGPU+NPU UMA fusion plan.
- `RLIMIT_MEMLOCK = unlimited` is required for the user (the limits.d
  drop-in `/etc/security/limits.d/90-kaden-memlock.conf` already does
  this).
- **Hugepages NOT required** with the OOT driver — it manages the
  IOMMU page-size internally. (Hugepages were a workaround the in-tree
  path needed and didn't even fully implement.)

## Verification

`/tmp/xdna-hwctx-roundtrip` (committed in this worktree) sweeps
`(num_columns ∈ {1,2,4,8}) × (max_opc ∈ {0,1,1024,65536,1048576}) ×
(log_buf ∈ {none, real})` = 40 combos.

Result on hipx with OOT driver: **40/40 winning combinations.** Every
hwctx creates cleanly, returns a real handle + syncobj, and destroys
without firmware errors. DEV_HEAP gets a valid `xdna_addr=0x4000000`
(was `0xffffffffffffffff` = INVALID_ADDR with the in-tree driver).

Additional shim_test verification (from the OOT repo's own test suite):

```
test #9 (create_invalid_bo)               PASSED
test #10 (create_and_free_exec_buf_bo)    PASSED   ← CMD BO path
test #53 (create and destroy devices)     PASSED
test #62 (create and free internal bo)    PASSED
test #65 (AIE MEM read/write)             FAILED  ← needs CU loaded
```

## Setup runbook (for future hipx-class machines)

See `scripts/setup-hipx-npu.sh`.

Quick version:
```bash
sudo apt install -y dkms build-essential cmake git linux-headers-$(uname -r) libelf-dev pkg-config jq
git clone --recursive https://github.com/amd/xdna-driver.git
cd xdna-driver/build
./build.sh -release
sudo dpkg -i --force-depends Release/xrt_plugin.*-amdxdna.deb

# memlock for the user (replace `kaden` with your user)
echo "kaden hard memlock unlimited
kaden soft memlock unlimited" | sudo tee /etc/security/limits.d/90-npu-memlock.conf

# log out + back in for memlock to take effect
ulimit -l   # should print "unlimited"
```

## What's next (Phase 1.3+)

The ioctl path is unblocked. The next gating dependency is **a CU
artifact (xclbin/PDI) compiled for AIE-2P / npu_7**. Three candidate
sources:

1. **Build MLIR-AIE 1.3 from source for AIE-2P** and compile a hello-
   world CU. Open path, no closed-source deps. This is the canonical
   plan from the strategy doc.
2. **Mine published examples** — github.com/amd/IRON, Riallto, public
   Triton-XDNA examples that target AIE-2P. May have a "passthrough"
   or "memcpy" CU we can use as the EXEC_CMD smoke test.
3. **Ride the OOT shim_test** — the test binary expects xclbins at
   `local_shim_test_data/npu5/1x4/1x4.xclbin`; AMD doesn't ship them
   in this repo but they're internally generated. Could be reproduced
   by following AMD's own AIE-2P hello-world build.

For Phase 1.3 the minimum we need is **one** working PDI — anything
that the firmware accepts as a valid CU. Once we can submit + wait on
a real EXEC_CMD, the whole pipeline (BO→hwctx→CONFIG_HWCTX→EXEC_CMD→
syncobj_wait) is exercised, and we can move to Phase 1.5 (real INT8
GEMM).
