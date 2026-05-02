// asym3_dequant_256_f32 — diagnostic variant of asym3_dequant_256.
// Output is 256 fp32 (1024 bytes) instead of 256 bf16, so we can see
// the pre-bf16-rounding mul accumulator. Used by the stage 1.1
// verifier to characterize AIE-2P bf16 mul semantics.

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
    aie.objectfifo @f32_out(%core, {%shim_out}, 2 : i32)
        : !aie.objectfifo<memref<256xf32>>

    func.func private @asym3_dequant_256_f32(memref<96xui8>, memref<1xf32>, memref<256xf32>) attributes {
      link_with = "asym3_dequant_kernel.o"
    }

    %0 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %iter = %c0 to %c_inf step %c1 {
        %p_view = aie.objectfifo.acquire @packed_in(Consume, 1)
            : !aie.objectfifosubview<memref<96xui8>>
        %p = aie.objectfifo.subview.access %p_view[0]
            : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>

        %c_view = aie.objectfifo.acquire @cnorm_in(Consume, 1)
            : !aie.objectfifosubview<memref<1xf32>>
        %c = aie.objectfifo.subview.access %c_view[0]
            : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>

        %o_view = aie.objectfifo.acquire @f32_out(Produce, 1)
            : !aie.objectfifosubview<memref<256xf32>>
        %o = aie.objectfifo.subview.access %o_view[0]
            : !aie.objectfifosubview<memref<256xf32>> -> memref<256xf32>

        func.call @asym3_dequant_256_f32(%p, %c, %o)
            : (memref<96xui8>, memref<1xf32>, memref<256xf32>) -> ()

        aie.objectfifo.release @packed_in(Consume, 1)
        aie.objectfifo.release @cnorm_in(Consume, 1)
        aie.objectfifo.release @f32_out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%packed: memref<96xui8>, %cnorm: memref<1xf32>, %out: memref<256xf32>) {
      %t_packed = aiex.dma_configure_task_for @packed_in {
        aie.dma_bd(%packed : memref<96xui8>, 0, 96,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 1, stride = 0>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed)

      %t_cnorm = aiex.dma_configure_task_for @cnorm_in {
        aie.dma_bd(%cnorm : memref<1xf32>, 0, 1,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 1, stride = 0>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm)

      %t_out = aiex.dma_configure_task_for @f32_out {
        aie.dma_bd(%out : memref<256xf32>, 0, 256,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 1, stride = 0>, <size = 256, stride = 1>])
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
