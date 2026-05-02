// asym3_score_one/aie.mlir - stage 2.6 single-(head, position)
// fused score kernel for AIE-2P. MVP shape: single core.
//
// AIE-2P compute tiles have only 2 input + 2 output DMA channels.
// To stay within that budget, all per-call inputs are concatenated
// host-side into a single 3200-byte buffer and accessed via typed
// pointer arithmetic in the C++ kernel. See asym3_score_kernel.cc
// for the layout.
//
//   input_in:  3200 bytes (packed + cnorm + centers + trig tables)
//   score_out:    4 bytes (1 f32)
//
// Multi-core fan-out + per-layer batching follow once correctness
// is proven via parity test against triattn_score_asym3.

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
      scf.for %iter = %c0 to %c_inf step %c1 {
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
      aie.end
    }

    aie.runtime_sequence(%input: memref<3200xui8>, %score: memref<1xf32>) {
      %t_input = aiex.dma_configure_task_for @input_in {
        aie.dma_bd(%input : memref<3200xui8>, 0, 3200,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 1, stride = 0>, <size = 3200, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_input)

      %t_score = aiex.dma_configure_task_for @score_out {
        aie.dma_bd(%score : memref<1xf32>, 0, 1,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 1, stride = 0>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_score)
      aiex.dma_await_task(%t_score)
      aiex.dma_free_task(%t_input)
    }
  }
}
