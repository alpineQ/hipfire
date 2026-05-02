// asym3_dequant_layer/aie.mlir — per-layer batched asym3 dequant.
//
// Single dispatch covers N_ITERS (head, position) pairs. Each
// iteration: pull 96 bytes packed + 1 f32 cnorm, push 256 bf16
// out. Compute is identical to asym3_dequant_256; only the host-
// side DMA descriptors and the core's outer loop scale up.
//
// Stage 1.4 MVP iteration: N_ITERS = 32. Production-shape work
// (full 27B Gemma layer at 8 heads x 4096 positions = 32768
// iters) follows in the multi-core scaling pass.
//
// Three ObjectFifos shuttle data shim<->core; each iteration
// dequeues one element from each:
//
//   packed_in:  shim -> core, N_ITERS x 96 bytes
//   cnorm_in:   shim -> core, N_ITERS x 4 bytes
//   bf16_out:   core -> shim, N_ITERS x 512 bytes

module {
  aie.device(npu2) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_packed = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_cnorm  = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out    = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.objectfifo @packed_in(%shim_packed, {%core}, 2 : i32)
        : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in(%shim_cnorm, {%core}, 2 : i32)
        : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out(%core, {%shim_out}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    func.func private @asym3_dequant_layer_one(memref<96xui8>, memref<1xf32>, memref<256xbf16>) attributes {
      link_with = "asym3_dequant_kernel.o"
    }

    %0 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_iters = arith.constant 32 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in(Consume, 1)
              : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0]
              : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>

          %c_view = aie.objectfifo.acquire @cnorm_in(Consume, 1)
              : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0]
              : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>

          %o_view = aie.objectfifo.acquire @bf16_out(Produce, 1)
              : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0]
              : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>

          func.call @asym3_dequant_layer_one(%p, %c, %o)
              : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()

          aie.objectfifo.release @packed_in(Consume, 1)
          aie.objectfifo.release @cnorm_in(Consume, 1)
          aie.objectfifo.release @bf16_out(Produce, 1)
        }
      }
      aie.end
    }

    // Host-side DMA orchestration. Three buffers sized for N_ITERS=32
    // (head, position) pairs:
    //   packed: 32 * 96  = 3072 bytes
    //   cnorm:  32 * 4   = 128 bytes
    //   out:    32 * 256 = 8192 bf16 = 16384 bytes
    //
    // Stride patterns express N_ITERS contiguous chunks. The
    // ObjectFifo size matches the per-iteration chunk size, so the
    // shim DMA produces N_ITERS ObjectFifo elements per task.
    aie.runtime_sequence(%packed: memref<3072xui8>, %cnorm: memref<32xf32>, %out: memref<8192xbf16>) {
      %t_packed = aiex.dma_configure_task_for @packed_in {
        aie.dma_bd(%packed : memref<3072xui8>, 0, 3072,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 32, stride = 96>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed)

      %t_cnorm = aiex.dma_configure_task_for @cnorm_in {
        aie.dma_bd(%cnorm : memref<32xf32>, 0, 32,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 32, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm)

      %t_out = aiex.dma_configure_task_for @bf16_out {
        aie.dma_bd(%out : memref<8192xbf16>, 0, 8192,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 32, stride = 256>, <size = 256, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_packed)
      aiex.dma_free_task(%t_cnorm)
    }
  }
}
