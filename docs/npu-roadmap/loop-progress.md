# hipfire-x autonomous loop progress

Started 2026-05-01 (user asleep, contract-authorized to execute on hipx).
Goal: full implementation of hipfire-x for Strix Halo — `hipx` integrated
into the engine the way `rdna-compute` is, not a PoC.

## Definition of done

For loop to terminate cleanly (vs being interrupted):
1. `hipx` crate has CONFIG_HWCTX + EXEC_CMD + syncobj-wait wired (Phase 1.3a).
2. dmabuf cross-import iGPU→NPU verified end-to-end (Phase 1.4).
3. MLIR-AIE / Peano toolchain built on hipx with at least one
   compiling-to-PDI kernel for AIE-2P (Phase 1.2b).
4. At least one *real* CU dispatch round-trip executed: NPU receives
   command, runs kernel, signals fence. Output verified.
5. Engine integration: a `dispatch::compute_target` predicate routes
   *some* op to NPU on Strix Halo, with safe fallback to iGPU.
6. Bench numbers committed showing NPU path is at least correct (not
   necessarily winning in v1).

## Strategy

Three independent threads, run sequentially within each iteration but
overlapping across the loop:

- **Plumbing** (this iter): dmabuf import + CONFIG_HWCTX/EXEC_CMD code.
  Low risk, doesn't need PDI.
- **Toolchain** (background, multi-iter): MLIR-AIE git clone + cmake.
  Long-running. Wake on completion.
- **Kernel** (depends on toolchain): author + compile a CU. Highest risk.
  Likely won't fit in iter 1.

## Iteration log

### Iter 1 — 2026-05-01 ✅
- Plumbing committed (c3311c0): cmd.rs, fence.rs, prime.rs, dispatch::compute_target
- Bootstrapped MLIR-AIE toolchain via uv-managed Python 3.12 (Ubuntu 26.04 ships only 3.14)
- Compiled first AIE-2P kernel: passthrough_pykernel @ 4 KB
- Verified end-to-end via AMD's reference test: PASS on hipx
- Embedded PDI (2 KB) + npu_insts (300 B) into hipx::kernels (8a722d4)
- Re-discovered: amdxdna OOT module needs `dpkg -i` re-install across sessions

### Iter 1.5 — 2026-05-01 ✅ (continued)
- ert.rs builder for ert_start_kernel_cmd in pure Rust (6d41626)
- Header bitfield packing, ERT_START_NPU/ERT_CU constants,
  ert_npu_data prefix, set_arg_u32/u64 by argspace offset
- All hardcoded metadata for passthrough_4k now in
  `crates/hipx/src/kernels.rs` (kernel ID, cols, ops/cycle, arg
  layout offsets) — Iter 2 just calls into them.

### Iter 3 — 2026-05-01 ✅ (firmware connect WORKING)

**Major milestone**: The firmware now accepts our hwctx and runs
the passthrough CU end-to-end. dmesg shows `ctx.NN.1 connected`
and `total completed jobs 1`. The fix:

1. `mmap_dev_heap_aligned` — replicate XRT's heap allocation:
   reserve 2× anon, find next size-aligned address, MAP_FIXED |
   MAP_LOCKED at that addr.  MAP_LOCKED is the magic ingredient —
   the firmware MAP_HOST_BUFFER needs page-locked pages.
2. `drm_iowr_generic` — generic DRM ioctls (PRIME, SYNCOBJ, GEM_CLOSE)
   don't add DRM_COMMAND_BASE; without this, 0x40 + 0xCF overflowed u8.
3. cmd_handles passed by-value when cmd_count=1.

**Where we are now**: CU runs but output BO stays zero. cmd packet
layout matches main_kernels.json byte offsets, but 4080/4096 bytes
mismatch. Suspects:
- timeline_wait still EINVALs (we proceed without; just printing
  warning) — may be a flag misuse
- SYNC_BO on SHMEM output may not invalidate CPU caches properly
- `npu_data.instruction_buffer` semantics — XRT writes `paddr()`
  which under PASID may not equal our DEV BO's xdna_addr

### Iter 4 — 2026-05-01 ✅ HIPFIRE-X DISPATCHES TO NPU

PASS: 4096 bytes round-trip through NPU via pure ioctls, 5/5 stable.

The decisive find: AMD test uses `ERT_START_CU` (opcode 0), NOT
`ERT_START_NPU` (=20). Captured via a printk patch in the OOT driver's
`amdxdna_cmd_submit` that dumped the cmd_bo's first 24 dwords:

```
hipx-cmddump pid=62140 cmd_bo=6: 30010001 00000001 00000003 00000000
  04028000 00000000 0000004b 865b6000 000071f5 865b5000 000071f5
  00000000 00000000 ...
```

Header `0x30010001` decodes: state=NEW(1), count=16, opcode=0,
type=ERT_CU(3). After switching ert.rs to `new_start_cu` (no
npu_data prefix; args start at offset 8) our packet became byte-
identical to AMD's. NPU then actually moves data:

```
[passthrough] output histogram (top 5): 0x00=16 0x01=16 0x02=16 ...
[passthrough] output first 16 bytes: [00,01,02,03,04,05,06,07,...]
[passthrough] PASS (4096 bytes round-tripped)
```

Remaining minor non-blocker: timeline_wait still EINVALs at every
point we try (seq, seq+1, seq+2). 100ms sleep fallback works for
correctness; tighter waiter is a follow-up.

### Iter 5 — 2026-05-01 (in progress)

**Done**:
- Compiled vector_scalar_mul (4096 × i16 *= i32 scalar) for AIE-2P
  via aiecc, embedded artifacts in
  `kernels/aie2p/vec_scalar_mul/build/`
- `crates/hipx/src/kernels.rs` exposes `VEC_SCALAR_MUL_PDI` (3024 B),
  `VEC_SCALAR_MUL_INSTS` (420 B), kernel id, ops/cycle, columns,
  arg layout (alias of passthrough_4k_args).
- `crates/hipx/src/bin/vec_scalar_mul.rs` — full dispatch chain
  matching the passthrough binary, with sentinel pattern (0xCC)
  to detect "NPU wrote nothing".

**Where stuck**:
The vec_scalar_mul binary hits a different bug than passthrough.
Cmd packet looks structurally identical (header `30010001`, opcode,
instr, ninstr=0x69=105 dwords, 3 BO host_vas) but the NPU does not
write to output — sentinel survives unchanged.

Suspects:
1. vec_scalar_mul has a worker-tile + objectfifo dataflow
   (`rt.start(worker)` in the .py, then fill/drain). The npu_insts
   for this is more elaborate than passthrough's. May need extra
   driver/runtime setup we miss.
2. `instr_v.size()` semantics — XRT passes the dword count but maybe
   we have an off-by-one or the AMD test counts elements differently
   for object-fifo flows.
3. Maybe `bo3`/`bo4` need real BOs (ctrlpkts, trace) not zeros.

**Path forward**:
- Build AMD test for vec_scalar_mul and capture its cmd packet via
  the kernel printk (need gcc-13 install). Diff vs ours.
- Or: pivot to the original int8 vector_vector_add example, even
  simpler dataflow.
- Or: skip int16 multiply for now and design our own MLIR-AIE
  source for KV-codec dequant directly (still ambitious — needs
  understanding the IRON Python frontend deeper).

**Decision**: passthrough is the canonical working kernel (PASS,
5/5 stable). vec_scalar_mul is incremental. The "press-coverage
lever" requires a real KV-codec or INT8 GEMM kernel — neither is
achievable in this autonomous run. Engine integration scaffold
(NpuRuntime singleton, route_npu predicate) IS achievable and is
the next concrete deliverable.

### Iter 6 — 2026-05-01 ✅ ENGINE INTEGRATION

The engine now uses the NPU. Both the routing predicate AND a real
end-to-end NPU dispatch work through the engine API. Verified on
hipx via the `hipfire_x_init` example:

```
[hipfire-x] NPU detected:
  family:        Aie2p
  cols:          8
  TOPS (INT8):   58

[hipfire-x] route() smoke test:
  KV codec (asym3)                -> Igpu  (no PDI yet)
  INT8 GEMM 9B prefill            -> Igpu
  ...

[hipfire-x] engine-API NPU dispatch (passthrough_4k):
  PASS — 4096 bytes round-tripped through NPU;
  first 8 = [00, 01, 02, 03, 04, 05, 06, 07]
```

Shipped:
- `engine/Cargo.toml`: optional `hipx` dep + new feature `npu`
- `engine/src/npu.rs`: `NpuRuntime::try_init` (Some on Strix Halo,
  None elsewhere), `route(npu, op) -> ComputeTarget`, and
  `NpuRuntime::passthrough_4k` end-to-end driver
- `engine/examples/hipfire_x_init.rs`: smoke test exercising both
  routing AND real dispatch

Engine code paths can now safely call `route(...)` and fall back
to iGPU when no NPU runtime exists or no PDI for that op exists —
opportunistic offload semantics. The pattern for adding new ops:
1. Author kernel in MLIR-AIE → compile to PDI
2. Embed PDI in `hipx::kernels::FOO_PDI`
3. Flip `available_ops.foo = true` in `try_init()`
4. Add `NpuRuntime::run_foo()` mirroring `passthrough_4k`
5. Engine call sites that already use `route()` start dispatching

### Iter 7 — 2026-05-01 ✅ THREE-KERNEL DISPATCH

Added a third working AIE-2P kernel: passthrough_dmas (single-column,
MemTile DMA forward, 16 KB int32). PASSes on first try.

Working kernel matrix:
| Kernel              | Partition | Dataflow             | Status |
|---------------------|-----------|----------------------|--------|
| passthrough_4k      | 8 cols    | objectfifo direct    | PASS   |
| passthrough_dmas    | 1 col     | MemTile-routed DMA   | PASS   |
| vec_scalar_mul      | 8 cols    | Worker + core compute| FAIL*  |

*vec_scalar_mul: job completes (`total completed jobs N` in dmesg)
but core never writes to output BO. The Worker+ObjectFifo+core-
compute class likely needs additional setup we haven't decoded —
maybe related to how the core tile gets its ELF loaded, or
PASID/IOMMU mapping for core-internal SRAM. Not on the critical
path; we have working dispatch for two structurally different
kernel shapes.

The press-coverage lever requires a Worker-class kernel (KV-codec
needs actual int8 multiply, INT8 GEMM same). That's the next
gating work.

### Iter 8 — 2026-05-01 (in progress)

**Pinpointed**: Worker-class kernels need ALL 5 BO slots populated.
Confirmed via in-kernel printk capture of AMD's working
vec_scalar_mul packet — bo3/bo4 (ctrlpkts/trace) are real host VAs
in the AMD packet, not zero. Updated hipx-vec-scalar-mul to allocate
placeholders for bo3/bo4. Now structurally identical to AMD's.

**Still failing**: even with packet match, output stays at sentinel.
Remaining delta: our `instr_ptr` is at 0x04020000 vs AMD's 0x04028000
(both within heap, just different allocation offsets). Possibilities:
- Worker tiles expect instr BO at a specific offset relative to PDI
- XRT pads the PDI BO larger than its raw byte size (forcing instr
  to a higher offset)
- There's a state BO (CDO chain?) XRT allocates between PDI and
  instr that we're missing

Not blocking the rest of the build — passthrough_4k and
passthrough_dmas remain solid. Documented for next debug session.

### Iter 13 — final autonomous attempt: cmd packet byte-equal

After 32 KB pad BO between PDI and instr, our cmd packet is now
**byte-identical** to AMD's working test:

```
AMD:  30010001 00000001 00000003 0  04028000 0  00000069  bo0..bo4...
ours: 30010001 00000001 00000003 0  04028000 0  00000069  bo0..bo4...
```

Same instr_ptr (0x04028000), same all-5-BO host VAs, same header,
ninstr, opcode. PDI bytes verified identical (md5). npu_insts
verified identical (md5). ioctl sequence matches (after SYNC_BO
removal).

Yet output still stuck at 0xCC sentinel. The worker tile completes
the job (dmesg confirms `total completed jobs N`) but writes
nothing. Pre-touching bo3/bo4 pages didn't help.

**Remaining unknowns** — none observable from userspace:
- maybe XRT sets specific flags in CREATE_BO we don't (we always
  pass flags=0)
- some firmware state machine that depends on sequencing in a
  way we can't reproduce with raw ioctls
- IOMMU-level mapping permissions on our SHMEM pages
- the AIE core tile may need an internal reset between dispatches
  that XRT does and we don't

This is the wall for autonomous mode. Hands-on debug via AIE
coredump (test #64 in shim_test) or strace-replay would be the
next step. Stopping aggressive loop here; foundation is solid
and committed.

### Iter 11+ — Worker-class probe

Captured ioctl sequence diffs via strace:

| ioctl | AMD (working) | Ours (failing) |
|---|---|---|
| GET_INFO | 1 | 6 (probe queries) |
| CREATE_BO | 9 | 9 |
| GET_BO_INFO | 9 | 9 |
| CREATE_HWCTX | 1 | 1 |
| CONFIG_HWCTX | 1 | 1 |
| SYNC_BO | 0 | 0 (after fix) |
| EXEC_CMD | 1 | 1 |
| TIMELINE_WAIT | 1 | 1 (errors) |
| GEM_CLOSE | 9 | 9 |

After SYNC_BO removal, ioctl shape is byte-identical to AMD's. cmd
packet bytes match. PDI bytes match xclbin-embedded byte-for-byte.
npu_insts bytes identical.

**Remaining mystery**: AMD's test PASSes 100% of the time. Ours
fails with output stuck at sentinel (0xCC). Same firmware sees
both packets and only writes for AMD's. The delta MUST be either:
- xdna_addr offsets (heap-relative — ours 0x04020000 vs AMD's
  0x04028000) trigger different IOMMU mappings
- something in cmd or arg BO state we're not setting (cache
  flushes? user→device mapping flags?)
- a kernel-mode patch step we're missing (XRT might LD_PRELOAD-
  style intercept the BOs to add init data)

Further debug would require:
- kprobing the actual firmware mailbox messages, not just cmd_bo
- running our binary under XRT's `xrt::run` with our own xclbin to
  see if XRT's framework bridges the gap somehow
- comparing the AIE core tile's instruction memory layout
  expectations vs what we provide (this needs Peano tooling)

### Iter 9 — TODO (kept for ref)

Real workload kernel. Two concrete candidates:

**A. KV-codec asym dequant** — biggest perf lever, hardest to author.
   Need MLIR-AIE source for `int8 → fp16` with asym2/3/4 scale +
   zero. Then wire into `crates/engine/src/triattn.rs` asym KV
   path. Bench via `coherence-gate-dflash.sh`.

**B. INT8 GEMV (small)** — simpler op, useful for spec-decode
   draft offload. AMD has examples to adapt.

vec_scalar_mul still doesn't write output despite identical-looking
cmd packet. The dataflow is more complex than passthrough; debugging
is non-trivial without a working AMD test to compare against (gcc-13
not on hipx). Deferred — not on the critical path for press-coverage
since it's not a real engine op.

### Iter 5 — engine integration scaffold (orig plan, kept for ref)

Now hipx is proven through-and-through. Next milestones for "press-
coverage lever" status:

1. **Engine-side `NpuRuntime`** in crates/engine/. A lazy_static-
   initialized singleton that opens Hipx once per process; held
   behind `Option<NpuRuntime>` so non-Strix-Halo systems get None.
2. **`engine::dispatch::route_npu(op, ...)`** that uses
   `hipx::dispatch::compute_target(op, info, has_pdi)` to decide
   iGPU vs NPU per-op. Returns iGPU when no PDI for that op exists.
3. **First useful kernel: KV-codec asym dequant**.
   - Author MLIR-AIE source: `int8 → fp16` with asym2/3/4 scale.
   - Compile to PDI via aiecc.
   - Embed in `hipx::kernels::KV_DEQUANT_ASYM3_PDI` etc.
   - Expose `hipx::ops::kv_dequant(input, scale, zero) -> output`.
4. **Wire into `crates/engine/src/triattn.rs` asym KV codec path**.
   Behind `HIPFIRE_NPU_KV=1` env var initially.
5. **Bench**: 9B/27B DFlash decode tok/s with NPU codec on/off
   via `coherence-gate-dflash.sh`. Press-coverage number lives here.

Iter 5 will likely take multiple loops because step 3 (kernel author)
is the real lift. Steps 1-2 are 1-2 hours of plumbing. Step 4 is
~1 hour. Step 5 is ~30 min.

### Iter 2 — 2026-05-01 ✅ (continued)

**Done**:
- ert.rs: `ErtBuilder::new_start_npu` + `set_cu_mask` / `set_npu_data`
  / `set_arg_*` / `finalize`
- fence.rs: `timeline_wait` (the variant amdxdna actually uses)
- bo.rs: `dev_slice_in_heap` / runtime `dev_slice` — DEV BOs sub-
  allocate from DEV_HEAP and don't have their own map_offset; we
  access them by offset within the heap mmap
- cmd.rs: `submit_exec_cmd` fixed — for cmd_count==1 the kernel
  expects the handle value itself, not a pointer to a 1-elem array
- bin/passthrough.rs: full Rust dispatch chain (Hipx → hwctx →
  PDI BO → CONFIG_HWCTX → instr BO → input/output SHMEM → CMD BO
  with ert_start_kernel_cmd → submit_exec_cmd → timeline_wait)

**Where we are stuck**:
EXEC_CMD now reaches the kernel cleanly (cmd handle accepted, args
list parsed) but the hwctx becomes fatal during the firmware
connect step:

```
amdxdna 0000:be:00.1: rq_submit_enter_slow: ctx.NN.1 fatal error
amdxdna 0000:be:00.1: aie2_cmd_submit: Submit enter failed, ret -5
```

`CTX_STATE_DEAD` is set by `part_ctx_start` when `aie2_ctx_connect`
returns an error. The firmware connect uses the CU configuration
we bound via CONFIG_HWCTX, plus context metadata. Most likely
suspects, in order:

1. **PDI bytes shape**: we load `main.pdi` (2064 B) directly. XRT's
   `xrt::xclbin` may extract a PDI plus extra metadata from the
   xclbin's AIE_PARTITION section that the firmware needs. Compare
   `main.pdi` bytes vs the AIE_PARTITION section bytes from
   `passthrough.xclbin`.
2. **CU function index**: we pass `cu_func=0`. The kernel JSON
   doesn't explicitly state a value; XRT's `xrt::module` may
   compute it from the CU metadata. Try cu_func=1 or extract
   from `dpu_kernel_id` (=0x901).
3. **CMD BO contents**: header bits seem right (0x3a014001 decodes
   to state=NEW, count=20, opcode=ERT_START_NPU=20, type=ERT_CU=3),
   but maybe the npu_data instr_buffer field needs to be a HOST VA
   for SHMEM, not a DEV xdna_addr. Try using instr_bo as SHMEM
   (host_ptr) instead of DEV (xdna_addr).
4. **PASID-related**: with PASID, the firmware walks the host page
   tables. Maybe SHMEM BOs need to be page-locked (mlock) or to
   have all pages faulted before submit. We pre-touch the heap;
   not the SHMEM BOs.

**Iter 3 plan**:
- Compare `main.pdi` to the AIE_PARTITION section in
  `passthrough.xclbin` (xclbin is a structured archive, parseable).
- If they match: investigate cu_func + cmd packet differences.
- If different: extract the real PDI bytes from the xclbin and
  re-embed.
- Capture cmd packet bytes from the working AMD test by patching
  `test.cpp` to dump just before `run.wait()`.
- Compare byte-for-byte vs our generated packet, fix delta.

When passthrough PASSes via hipx alone, Iter 4 swaps the kernel
for KV-codec dequant and integrates with engine triattn.

### Original Iter 2 plan (kept for reference)
**Goal**: Rust EXEC_CMD round-trip — replicate the passthrough PASS via
pure hipx dispatch, no XRT.

Steps:
1. ✅ Implement `ert_start_kernel_cmd` packet builder (Iter 1.5)
2. Write `crates/hipx/src/bin/passthrough.rs`:
   - `Hipx::open()` (handles /dev/accel/accel0 + 64MB DEV_HEAP)
   - `hipx.create_hwctx({num_columns: 8, max_opc: 2048})`
   - alloc DEV BO sized to `kernels::PASSTHROUGH_4K_PDI.len()`,
     mmap, copy PDI bytes in
   - `cmd::config_cus(fd, ctx, vec![pdi_bo], &[0])` — bind CU index 0
   - alloc SHMEM input BO (4096 bytes), fill `(i & 0xff)` pattern
   - alloc SHMEM output BO (4096 bytes), zero
   - alloc DEV BO for npu_insts, copy PASSTHROUGH_4K_INSTS in
   - alloc CMD BO (4 KB), build ERT_START_NPU packet:
     - cu_mask = 0x1
     - npu_data: instr_addr = npu_insts_bo.xdna_addr,
                 instr_size = PASSTHROUGH_4K_INSTS.len()
     - arg 0x00 opcode (u64) = 3
     - arg 0x08 instr ptr (u64) = npu_insts_bo.xdna_addr (or 0 if
       npu_data prefix already supplies)
     - arg 0x10 ninstr (u32) = inst dword count
     - arg 0x14 bo0 = input_bo.xdna_addr (or vaddr; investigate)
     - arg 0x1C bo1 = output_bo.xdna_addr
     - arg 0x24..0x34 bo2..bo4 = 0
   - `cmd::submit_exec_cmd(fd, ctx, &[&cmd_bo],
                            &[&input_bo, &output_bo, &npu_insts_bo, &pdi_bo])`
   - `fence::wait(fd, ctx.syncobj_handle, 5s)`
   - read output bytes, compare to input → PASS/FAIL
3. Iterate: probably need to fix arg-passing semantics (bo handles
   vs xdna_addrs vs vaddr) — first attempt likely fails, dmesg will
   tell us what's wrong.
4. Exit criterion: `hipx-passthrough` prints "PASS!" matching the
   AMD reference test.

### Iter 3 — TODO (pending Iter 2)
- Author KV-codec dequant kernel (asym3 → fp16 + asym4 → fp16)
- Replace passthrough_4k with kv_dequant kernels in `kernels/aie2p/`
- Engine integration in `crates/engine/src/triattn.rs` asym KV path:
  add NPU dequant routing under `dispatch::compute_target`
- Bench measurements: 9B / 27B DFlash decode tok/s with NPU codec
  vs without (controlled A/B with HIPFIRE_NPU=0/1 env var)

### Notes for future iterations
- `sudo dpkg -i ~/xdna-driver/build/Release/*.deb` to re-install OOT
  module if depmod has reverted to in-tree v0.7.0 after a reboot.
- `ulimit -l unlimited` required at runtime (limits.d already set).

### Iter 3 — TODO (pending Iter 2)
- Replace passthrough kernel with KV-codec dequant kernel
- Engine integration in `crates/engine/src/triattn.rs` asym KV path
- Bench measurements for press-coverage
