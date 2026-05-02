// asym3_score_layer/aie.mlir - stage 2.6 multi-iter fused score
// kernel for AIE-2P. Single core, N_ITERS chunks per dispatch.
//
// Each iteration consumes a 3200-byte input chunk (same layout as
// asym3_score_one) and produces 1 f32 score. The kernel C++
// (asym3_score_kernel.cc) is unchanged from the single-iter
// variant; multi-iter is host-side: bigger DMA, more ObjectFifo
// cycles in the core scf.for loop.
//
// N_ITERS = 128 for this scaling test. Per-dispatch:
//   input_in:   N_ITERS * 3200 = 409600 bytes
//   score_out:  N_ITERS * 4    =    512 bytes

module {
  aie.device(npu2) {
    %core         = aie.logical_tile<CoreTile>(?, ?)
    %shim_input   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_score   = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.objectfifo @input_in(%shim_input, {%core}, 2 : i32)
        : !aie.objectfifo<memref<3200xui8>>
    aie.objectfifo @score_out(%core, {%shim_score}, 2 : i32)
        : !aie.objectfifo<memref<1xf32>>

    func.func private @asym3_score_one(memref<3200xui8>, memref<1xf32>) attributes {
      link_with = "asym3_score_kernel.o"
    }

    %0 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_iters = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters step %c1 {
          %i_view = aie.objectfifo.acquire @input_in(Consume, 1)
              : !aie.objectfifosubview<memref<3200xui8>>
          %i = aie.objectfifo.subview.access %i_view[0]
              : !aie.objectfifosubview<memref<3200xui8>> -> memref<3200xui8>
          %s_view = aie.objectfifo.acquire @score_out(Produce, 1)
              : !aie.objectfifosubview<memref<1xf32>>
          %s = aie.objectfifo.subview.access %s_view[0]
              : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>

          func.call @asym3_score_one(%i, %s)
              : (memref<3200xui8>, memref<1xf32>) -> ()

          aie.objectfifo.release @input_in(Consume, 1)
          aie.objectfifo.release @score_out(Produce, 1)
        }
      }
      aie.end
    }

    aie.runtime_sequence(%input: memref<409600xui8>, %score: memref<128xf32>) {
      // Single big DMA: 128 contiguous chunks of 3200 bytes each.
      %t_input = aiex.dma_configure_task_for @input_in {
        aie.dma_bd(%input : memref<409600xui8>, 0, 409600,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 128, stride = 3200>, <size = 3200, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_input)

      %t_score = aiex.dma_configure_task_for @score_out {
        aie.dma_bd(%score : memref<128xf32>, 0, 128,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 128, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_score)
      aiex.dma_await_task(%t_score)
      aiex.dma_free_task(%t_input)
    }
  }
}
