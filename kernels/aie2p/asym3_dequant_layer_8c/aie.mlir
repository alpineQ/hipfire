// asym3_dequant_layer_8c/aie.mlir - 8-core full-column kernel.
//
// Uses all 8 columns of the Strix Halo NPU2. Each column has its
// own {shim_noc, compute_tile} pair handling N_ITERS/8 = 128 chunks.
// Same independent-column pattern as 4c, just doubled.
//
// Per-column buffer slices (N_ITERS=1024):
//   packed: 128 * 96  = 12288 B; offsets 0, 12288, 24576, ..., 86016
//   cnorm:  128 * 4   =   512 B; offsets 0, 128, 256, ..., 896 (f32 elems)
//   out:    128 * 256 = 32768 bf16 elems; offsets 0, 32768, ..., 229376

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_4_0 = aie.tile(4, 0)
    %tile_5_0 = aie.tile(5, 0)
    %tile_6_0 = aie.tile(6, 0)
    %tile_7_0 = aie.tile(7, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)

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
    aie.objectfifo @packed_in_4(%tile_4_0, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_4(%tile_4_0, {%tile_4_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_4(%tile_4_2, {%tile_4_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo @packed_in_5(%tile_5_0, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_5(%tile_5_0, {%tile_5_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_5(%tile_5_2, {%tile_5_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo @packed_in_6(%tile_6_0, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_6(%tile_6_0, {%tile_6_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_6(%tile_6_2, {%tile_6_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.objectfifo @packed_in_7(%tile_7_0, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in_7(%tile_7_0, {%tile_7_2}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @bf16_out_7(%tile_7_2, {%tile_7_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>

    func.func private @asym3_dequant_layer_one(memref<96xui8>, memref<1xf32>, memref<256xbf16>) attributes {
      link_with = "asym3_dequant_kernel.o"
    }

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
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
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
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
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
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
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
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
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_4(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_4(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_4(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_4(Consume, 1)
          aie.objectfifo.release @cnorm_in_4(Consume, 1)
          aie.objectfifo.release @bf16_out_4(Produce, 1)
        }
      }
      aie.end
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_5(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_5(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_5(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_5(Consume, 1)
          aie.objectfifo.release @cnorm_in_5(Consume, 1)
          aie.objectfifo.release @bf16_out_5(Produce, 1)
        }
      }
      aie.end
    }
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_6(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_6(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_6(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_6(Consume, 1)
          aie.objectfifo.release @cnorm_in_6(Consume, 1)
          aie.objectfifo.release @bf16_out_6(Produce, 1)
        }
      }
      aie.end
    }
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c_per_col = arith.constant 128 : index
      scf.for %outer = %c0 to %c_inf step %c1 {
        scf.for %iter = %c0 to %c_per_col step %c1 {
          %p_view = aie.objectfifo.acquire @packed_in_7(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
          %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
          %c_view = aie.objectfifo.acquire @cnorm_in_7(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
          %c = aie.objectfifo.subview.access %c_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
          %o_view = aie.objectfifo.acquire @bf16_out_7(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
          %o = aie.objectfifo.subview.access %o_view[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
          func.call @asym3_dequant_layer_one(%p, %c, %o) : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()
          aie.objectfifo.release @packed_in_7(Consume, 1)
          aie.objectfifo.release @cnorm_in_7(Consume, 1)
          aie.objectfifo.release @bf16_out_7(Produce, 1)
        }
      }
      aie.end
    }

    aie.runtime_sequence(%packed: memref<98304xui8>, %cnorm: memref<1024xf32>, %out: memref<262144xbf16>) {
      // Per-column DMAs (8 cols, 128 chunks each, 1024 total).
      %t_packed_0 = aiex.dma_configure_task_for @packed_in_0 {
        aie.dma_bd(%packed : memref<98304xui8>, 0, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_0)
      %t_packed_1 = aiex.dma_configure_task_for @packed_in_1 {
        aie.dma_bd(%packed : memref<98304xui8>, 12288, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_1)
      %t_packed_2 = aiex.dma_configure_task_for @packed_in_2 {
        aie.dma_bd(%packed : memref<98304xui8>, 24576, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_2)
      %t_packed_3 = aiex.dma_configure_task_for @packed_in_3 {
        aie.dma_bd(%packed : memref<98304xui8>, 36864, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_3)
      %t_packed_4 = aiex.dma_configure_task_for @packed_in_4 {
        aie.dma_bd(%packed : memref<98304xui8>, 49152, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_4)
      %t_packed_5 = aiex.dma_configure_task_for @packed_in_5 {
        aie.dma_bd(%packed : memref<98304xui8>, 61440, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_5)
      %t_packed_6 = aiex.dma_configure_task_for @packed_in_6 {
        aie.dma_bd(%packed : memref<98304xui8>, 73728, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_6)
      %t_packed_7 = aiex.dma_configure_task_for @packed_in_7 {
        aie.dma_bd(%packed : memref<98304xui8>, 86016, 12288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 96>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed_7)

      %t_cnorm_0 = aiex.dma_configure_task_for @cnorm_in_0 { aie.dma_bd(%cnorm : memref<1024xf32>,    0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_0)
      %t_cnorm_1 = aiex.dma_configure_task_for @cnorm_in_1 { aie.dma_bd(%cnorm : memref<1024xf32>,  128, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_1)
      %t_cnorm_2 = aiex.dma_configure_task_for @cnorm_in_2 { aie.dma_bd(%cnorm : memref<1024xf32>,  256, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_2)
      %t_cnorm_3 = aiex.dma_configure_task_for @cnorm_in_3 { aie.dma_bd(%cnorm : memref<1024xf32>,  384, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_3)
      %t_cnorm_4 = aiex.dma_configure_task_for @cnorm_in_4 { aie.dma_bd(%cnorm : memref<1024xf32>,  512, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_4)
      %t_cnorm_5 = aiex.dma_configure_task_for @cnorm_in_5 { aie.dma_bd(%cnorm : memref<1024xf32>,  640, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_5)
      %t_cnorm_6 = aiex.dma_configure_task_for @cnorm_in_6 { aie.dma_bd(%cnorm : memref<1024xf32>,  768, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_6)
      %t_cnorm_7 = aiex.dma_configure_task_for @cnorm_in_7 { aie.dma_bd(%cnorm : memref<1024xf32>,  896, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm_7)

      %t_out_0 = aiex.dma_configure_task_for @bf16_out_0 { aie.dma_bd(%out : memref<262144xbf16>,      0, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_0)
      %t_out_1 = aiex.dma_configure_task_for @bf16_out_1 { aie.dma_bd(%out : memref<262144xbf16>,  32768, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_1)
      %t_out_2 = aiex.dma_configure_task_for @bf16_out_2 { aie.dma_bd(%out : memref<262144xbf16>,  65536, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_2)
      %t_out_3 = aiex.dma_configure_task_for @bf16_out_3 { aie.dma_bd(%out : memref<262144xbf16>,  98304, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_3)
      %t_out_4 = aiex.dma_configure_task_for @bf16_out_4 { aie.dma_bd(%out : memref<262144xbf16>, 131072, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_4)
      %t_out_5 = aiex.dma_configure_task_for @bf16_out_5 { aie.dma_bd(%out : memref<262144xbf16>, 163840, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_5)
      %t_out_6 = aiex.dma_configure_task_for @bf16_out_6 { aie.dma_bd(%out : memref<262144xbf16>, 196608, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_6)
      %t_out_7 = aiex.dma_configure_task_for @bf16_out_7 { aie.dma_bd(%out : memref<262144xbf16>, 229376, 32768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 256>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out_7)

      aiex.dma_await_task(%t_out_0)
      aiex.dma_await_task(%t_out_1)
      aiex.dma_await_task(%t_out_2)
      aiex.dma_await_task(%t_out_3)
      aiex.dma_await_task(%t_out_4)
      aiex.dma_await_task(%t_out_5)
      aiex.dma_await_task(%t_out_6)
      aiex.dma_await_task(%t_out_7)
      aiex.dma_free_task(%t_packed_0)
      aiex.dma_free_task(%t_packed_1)
      aiex.dma_free_task(%t_packed_2)
      aiex.dma_free_task(%t_packed_3)
      aiex.dma_free_task(%t_packed_4)
      aiex.dma_free_task(%t_packed_5)
      aiex.dma_free_task(%t_packed_6)
      aiex.dma_free_task(%t_packed_7)
      aiex.dma_free_task(%t_cnorm_0)
      aiex.dma_free_task(%t_cnorm_1)
      aiex.dma_free_task(%t_cnorm_2)
      aiex.dma_free_task(%t_cnorm_3)
      aiex.dma_free_task(%t_cnorm_4)
      aiex.dma_free_task(%t_cnorm_5)
      aiex.dma_free_task(%t_cnorm_6)
      aiex.dma_free_task(%t_cnorm_7)
    }
  }
}
