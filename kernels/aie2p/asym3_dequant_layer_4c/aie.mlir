// asym3_dequant_layer_4c/aie.mlir - 4-core multi-column kernel.
//
// Topology: four independent {shim_noc, compute_tile} pairs in
// columns 0-3, each handling N_ITERS/4 = 256 chunks. Same
// independent-column pattern as the 2-core variant; just scaled
// to four columns. No mem_tile staging; shim DMA goes directly
// to each column's compute tile L1 ObjectFifo.
//
//   shim(0,0) <-> compute(0,2): chunks   0..255
//   shim(1,0) <-> compute(1,2): chunks 256..511
//   shim(2,0) <-> compute(2,2): chunks 512..767
//   shim(3,0) <-> compute(3,2): chunks 768..1023
//
// Buffer sizes (N_ITERS=1024):
//   packed total: 1024 * 96  = 98304 B; per-col 24576 B (256 chunks)
//   cnorm  total: 1024 * 4   =  4096 B; per-col  1024 B (256 f32)
//   out    total: 1024 * 512 = 524288 B; per-col 131072 B (256 chunks)

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)

    aie.objectfifo @packed_in_0(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_0(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_0(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>

    aie.objectfifo @packed_in_1(%tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_1(%tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_1(%tile_1_2, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>

    aie.objectfifo @packed_in_2(%tile_2_0, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_2(%tile_2_0, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_2(%tile_2_2, {%tile_2_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>

    aie.objectfifo @packed_in_3(%tile_3_0, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_3(%tile_3_0, {%tile_3_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_3(%tile_3_2, {%tile_3_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>

    func.func private @asym3_dequant_layer_one(memref<96xui8>, memref<1xf32>, memref<256xbf16>) attributes {
      link_with = "asym3_dequant_kernel.o"
    }

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_iters_quarter = arith.constant 256 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters_quarter step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_0(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_0(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_0(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
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
      %c_iters_quarter = arith.constant 256 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters_quarter step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_1(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_1(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_1(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_1(Consume, 1)
          aie.objectfifo.release @cnorm_in_1(Consume, 1)
          aie.objectfifo.release @bf16_out_1(Produce, 1)
        }
      }
      aie.end
    }

    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_iters_quarter = arith.constant 256 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters_quarter step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_2(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_2(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_2(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_2(Consume, 1)
          aie.objectfifo.release @cnorm_in_2(Consume, 1)
          aie.objectfifo.release @bf16_out_2(Produce, 1)
        }
      }
      aie.end
    }

    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_iters_quarter = arith.constant 256 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_iters_quarter step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_3(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_3(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_3(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_3(Consume, 1)
          aie.objectfifo.release @cnorm_in_3(Consume, 1)
          aie.objectfifo.release @bf16_out_3(Produce, 1)
        }
      }
      aie.end
    }

    aie.runtime_sequence(%packed: memref<98304xui8>, %cnorm: memref<1024xf32>, %out: memref<262144xbf16>) {
      // Per-column input DMAs at offsets 0, 24576, 49152, 73728 (packed)
      // and 0, 256, 512, 768 (cnorm). Each picks 256 chunks.
      %t_packed_0 = aiex.dma_configure_task_for @packed_in_0 {
        aie.dma_bd(%packed : memref<98304xui8>, 0, 24576,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 96>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_0)

      %t_packed_1 = aiex.dma_configure_task_for @packed_in_1 {
        aie.dma_bd(%packed : memref<98304xui8>, 24576, 24576,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 96>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_1)

      %t_packed_2 = aiex.dma_configure_task_for @packed_in_2 {
        aie.dma_bd(%packed : memref<98304xui8>, 49152, 24576,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 96>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_2)

      %t_packed_3 = aiex.dma_configure_task_for @packed_in_3 {
        aie.dma_bd(%packed : memref<98304xui8>, 73728, 24576,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 96>, <size = 96, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_3)

      %t_cnorm_0 = aiex.dma_configure_task_for @cnorm_in_0 {
        aie.dma_bd(%cnorm : memref<1024xf32>, 0, 256,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_0)

      %t_cnorm_1 = aiex.dma_configure_task_for @cnorm_in_1 {
        aie.dma_bd(%cnorm : memref<1024xf32>, 256, 256,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_1)

      %t_cnorm_2 = aiex.dma_configure_task_for @cnorm_in_2 {
        aie.dma_bd(%cnorm : memref<1024xf32>, 512, 256,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_2)

      %t_cnorm_3 = aiex.dma_configure_task_for @cnorm_in_3 {
        aie.dma_bd(%cnorm : memref<1024xf32>, 768, 256,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 1>, <size = 1, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_3)

      // Per-column output DMAs at offsets 0, 65536, 131072, 196608
      // (bf16 elements). All issue tokens; all are awaited so the
      // host doesn't return before any column's data lands.
      %t_out_0 = aiex.dma_configure_task_for @bf16_out_0 {
        aie.dma_bd(%out : memref<262144xbf16>, 0, 65536,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 256>, <size = 256, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_0)

      %t_out_1 = aiex.dma_configure_task_for @bf16_out_1 {
        aie.dma_bd(%out : memref<262144xbf16>, 65536, 65536,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 256>, <size = 256, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_1)

      %t_out_2 = aiex.dma_configure_task_for @bf16_out_2 {
        aie.dma_bd(%out : memref<262144xbf16>, 131072, 65536,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 256>, <size = 256, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_2)

      %t_out_3 = aiex.dma_configure_task_for @bf16_out_3 {
        aie.dma_bd(%out : memref<262144xbf16>, 196608, 65536,
                   [<size = 1, stride = 0>, <size = 1, stride = 0>,
                    <size = 256, stride = 256>, <size = 256, stride = 1>])
                  {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_3)

      aiex.dma_await_task(%t_out_0)
      aiex.dma_await_task(%t_out_1)
      aiex.dma_await_task(%t_out_2)
      aiex.dma_await_task(%t_out_3)
      aiex.dma_free_task(%t_packed_0)
      aiex.dma_free_task(%t_packed_1)
      aiex.dma_free_task(%t_packed_2)
      aiex.dma_free_task(%t_packed_3)
      aiex.dma_free_task(%t_cnorm_0)
      aiex.dma_free_task(%t_cnorm_1)
      aiex.dma_free_task(%t_cnorm_2)
      aiex.dma_free_task(%t_cnorm_3)
    }
  }
}
