// asym3_dequant_layer_2c/aie.mlir - 2-core MVP of the multi-core
// layer dequant kernel. Phase A.
//
// Topology: two independent {shim_noc, compute_tile} pairs in
// adjacent columns, each handling N_ITERS/2 = 512 chunks.
// No mem_tile staging; shim DMA goes directly to each compute tile's
// L1 ObjectFifo. This is the simplest possible multi-core layout
// since each iteration is data-independent.
//
//   shim_noc(0,0) <-> compute(0,2): chunks 0..511
//   shim_noc(1,0) <-> compute(1,2): chunks 512..1023
//
// Host runtime sequence issues 2 input DMAs and 2 output DMAs,
// each at the offset for its column's share. The output read-back
// at the host side gets a contiguous 1024-chunk bf16 K (each
// column writes its half to the host buffer at the appropriate
// offset).

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)   // shim col 0
    %tile_1_0 = aie.tile(1, 0)   // shim col 1
    %tile_0_2 = aie.tile(0, 2)   // compute col 0 row 2
    %tile_1_2 = aie.tile(1, 2)   // compute col 1 row 2

    // Column 0 ObjectFifos (chunks 0..511).
    aie.objectfifo @packed_in_0(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_0(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_0(%tile_0_2, {%tile_0_0}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    // Column 1 ObjectFifos (chunks 512..1023).
    aie.objectfifo @packed_in_1(%tile_1_0, {%tile_1_2}, 2 : i32)
        : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_1(%tile_1_0, {%tile_1_2}, 2 : i32)
        : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_1(%tile_1_2, {%tile_1_0}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    func.func private @asym3_dequant_layer_one(memref<96xui8>, memref<1xf32>, memref<256xbf16>) attributes {
      link_with = "asym3_dequant_kernel.o"
    }

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_iters_half = arith.constant 512 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters_half step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_0(Consume, 1)
              : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0]
              : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_0(Consume, 1)
              : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0]
              : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_0(Produce, 1)
              : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0]
              : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o)
              : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_0(Consume, 1)
          aie.objectfifo.release @cnorm_in_0(Consume, 1)
          aie.objectfifo.release @bf16_out_0(Produce, 1)
        }
      }
      aie.end
    }

    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_iters_half = arith.constant 512 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters_half step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_1(Consume, 1)
              : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0]
              : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_1(Consume, 1)
              : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0]
              : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_1(Produce, 1)
              : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0]
              : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o)
              : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_1(Consume, 1)
          aie.objectfifo.release @cnorm_in_1(Consume, 1)
          aie.objectfifo.release @bf16_out_1(Produce, 1)
        }
      }
      aie.end
    }

    // Host runtime sequence: 2 input DMAs (per column) + 2 output
    // DMAs. The single packed/cnorm/out memref carries all 1024
    // chunks contiguously; each shim DMA picks its half via the
    // initial offset (0 for col 0, half-size for col 1).
    //
    // packed total: 1024 * 96  = 98304 B; per-col 49152 B at chunks
    //   of 96 B each (size = 512).
    // cnorm  total: 1024 * 4   = 4096 B; per-col 2048 B (size = 512
    //   chunks of 1 f32).
    // out    total: 1024 * 512 = 524288 B; per-col 262144 B
    //   (size = 512 chunks of 256 bf16).
    aie.runtime_sequence(%packed: memref<98304xui8>, %cnorm: memref<1024xf32>, %out: memref<262144xbf16>) {
      // Column 0 input DMAs.
      %t_packed_0 = aiex.dma_configure_task_for @packed_in_0 {
        aie.dma_bd(%packed : memref<98304xui8>, 0, 49152,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 512, stride = 96>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_0)

      %t_cnorm_0 = aiex.dma_configure_task_for @cnorm_in_0 {
        aie.dma_bd(%cnorm : memref<1024xf32>, 0, 512,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 512, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_0)

      // Column 1 input DMAs (offset: 49152 B for packed, 512 elems
      // for cnorm).
      %t_packed_1 = aiex.dma_configure_task_for @packed_in_1 {
        aie.dma_bd(%packed : memref<98304xui8>, 49152, 49152,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 512, stride = 96>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_1)

      %t_cnorm_1 = aiex.dma_configure_task_for @cnorm_in_1 {
        aie.dma_bd(%cnorm : memref<1024xf32>, 512, 512,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 512, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_1)

      // Output DMAs. Column 0 writes chunks 0..511 (offset 0,
      // 131072 bf16 elements); column 1 writes chunks 512..1023
      // (offset 131072 bf16 elements). BOTH issue completion
      // tokens and BOTH are awaited; awaiting only one lets the
      // host return before the other column's data lands.
      %t_out_0 = aiex.dma_configure_task_for @bf16_out_0 {
        aie.dma_bd(%out : memref<262144xbf16>, 0, 131072,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 512, stride = 256>, <size = 256, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_0)

      %t_out_1 = aiex.dma_configure_task_for @bf16_out_1 {
        aie.dma_bd(%out : memref<262144xbf16>, 131072, 131072,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 512, stride = 256>, <size = 256, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_1)
      aiex.dma_await_task(%t_out_0)
      aiex.dma_await_task(%t_out_1)
      aiex.dma_free_task(%t_packed_0)
      aiex.dma_free_task(%t_cnorm_0)
      aiex.dma_free_task(%t_packed_1)
      aiex.dma_free_task(%t_cnorm_1)
    }
  }
}
