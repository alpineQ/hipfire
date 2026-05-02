// asym3_score_one/aie.mlir - stage 2.6 single-(head, position)
// fused score kernel for AIE-2P. MVP shape: single core, single
// dispatch covers ONE (head, pos) chunk. Multi-core fan-out + per-
// layer batching follow once correctness is proven via parity test
// against the iGPU triattn_score_asym3 reference.
//
// Eight input ObjectFifos plus one output ObjectFifo:
//   packed_in:    96 bytes  (32 threads x 3 packed indices)
//   cnorm_in:      4 bytes  (1 f32)
//   c_mag_in:    512 bytes  (128 f32 per band)
//   c_abs_in:    512 bytes  (128 f32)
//   cos_a_in:    512 bytes  (128 f32 host-precomputed cos)
//   sin_a_in:    512 bytes  (128 f32 host-precomputed sin)
//   cos_t_in:    512 bytes  (128 f32 Givens cos)
//   sin_t_in:    512 bytes  (128 f32 Givens sin)
//   score_out:     4 bytes  (1 f32)

module {
  aie.device(npu2) {
    %core         = aie.logical_tile<CoreTile>(?, ?)
    %shim_packed  = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_cnorm   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_c_mag   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_c_abs   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_cos_a   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_sin_a   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_cos_t   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_sin_t   = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_score   = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.objectfifo @packed_in(%shim_packed, {%core}, 2 : i32) : !aie.objectfifo<memref<96xui8>>
    aie.objectfifo @cnorm_in(%shim_cnorm, {%core}, 2 : i32)   : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @c_mag_in(%shim_c_mag, {%core}, 2 : i32)   : !aie.objectfifo<memref<128xf32>>
    aie.objectfifo @c_abs_in(%shim_c_abs, {%core}, 2 : i32)   : !aie.objectfifo<memref<128xf32>>
    aie.objectfifo @cos_a_in(%shim_cos_a, {%core}, 2 : i32)   : !aie.objectfifo<memref<128xf32>>
    aie.objectfifo @sin_a_in(%shim_sin_a, {%core}, 2 : i32)   : !aie.objectfifo<memref<128xf32>>
    aie.objectfifo @cos_t_in(%shim_cos_t, {%core}, 2 : i32)   : !aie.objectfifo<memref<128xf32>>
    aie.objectfifo @sin_t_in(%shim_sin_t, {%core}, 2 : i32)   : !aie.objectfifo<memref<128xf32>>
    aie.objectfifo @score_out(%core, {%shim_score}, 2 : i32)  : !aie.objectfifo<memref<1xf32>>

    func.func private @asym3_score_one(
      memref<96xui8>, memref<1xf32>,
      memref<128xf32>, memref<128xf32>,
      memref<128xf32>, memref<128xf32>,
      memref<128xf32>, memref<128xf32>,
      memref<1xf32>) attributes {
      link_with = "asym3_score_kernel.o"
    }

    %0 = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c_inf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %iter = %c0 to %c_inf step %c1 {
        %p_view = aie.objectfifo.acquire @packed_in(Consume, 1) : !aie.objectfifosubview<memref<96xui8>>
        %p = aie.objectfifo.subview.access %p_view[0] : !aie.objectfifosubview<memref<96xui8>> -> memref<96xui8>
        %cn_view = aie.objectfifo.acquire @cnorm_in(Consume, 1) : !aie.objectfifosubview<memref<1xf32>>
        %cn = aie.objectfifo.subview.access %cn_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        %cm_view = aie.objectfifo.acquire @c_mag_in(Consume, 1) : !aie.objectfifosubview<memref<128xf32>>
        %cm = aie.objectfifo.subview.access %cm_view[0] : !aie.objectfifosubview<memref<128xf32>> -> memref<128xf32>
        %ca_view = aie.objectfifo.acquire @c_abs_in(Consume, 1) : !aie.objectfifosubview<memref<128xf32>>
        %ca = aie.objectfifo.subview.access %ca_view[0] : !aie.objectfifosubview<memref<128xf32>> -> memref<128xf32>
        %coa_view = aie.objectfifo.acquire @cos_a_in(Consume, 1) : !aie.objectfifosubview<memref<128xf32>>
        %coa = aie.objectfifo.subview.access %coa_view[0] : !aie.objectfifosubview<memref<128xf32>> -> memref<128xf32>
        %sia_view = aie.objectfifo.acquire @sin_a_in(Consume, 1) : !aie.objectfifosubview<memref<128xf32>>
        %sia = aie.objectfifo.subview.access %sia_view[0] : !aie.objectfifosubview<memref<128xf32>> -> memref<128xf32>
        %cot_view = aie.objectfifo.acquire @cos_t_in(Consume, 1) : !aie.objectfifosubview<memref<128xf32>>
        %cot = aie.objectfifo.subview.access %cot_view[0] : !aie.objectfifosubview<memref<128xf32>> -> memref<128xf32>
        %sit_view = aie.objectfifo.acquire @sin_t_in(Consume, 1) : !aie.objectfifosubview<memref<128xf32>>
        %sit = aie.objectfifo.subview.access %sit_view[0] : !aie.objectfifosubview<memref<128xf32>> -> memref<128xf32>
        %s_view = aie.objectfifo.acquire @score_out(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %s = aie.objectfifo.subview.access %s_view[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>

        func.call @asym3_score_one(%p, %cn, %cm, %ca, %coa, %sia, %cot, %sit, %s) : (
          memref<96xui8>, memref<1xf32>,
          memref<128xf32>, memref<128xf32>,
          memref<128xf32>, memref<128xf32>,
          memref<128xf32>, memref<128xf32>,
          memref<1xf32>) -> ()

        aie.objectfifo.release @packed_in(Consume, 1)
        aie.objectfifo.release @cnorm_in(Consume, 1)
        aie.objectfifo.release @c_mag_in(Consume, 1)
        aie.objectfifo.release @c_abs_in(Consume, 1)
        aie.objectfifo.release @cos_a_in(Consume, 1)
        aie.objectfifo.release @sin_a_in(Consume, 1)
        aie.objectfifo.release @cos_t_in(Consume, 1)
        aie.objectfifo.release @sin_t_in(Consume, 1)
        aie.objectfifo.release @score_out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(
      %packed: memref<96xui8>, %cnorm: memref<1xf32>,
      %c_mag: memref<128xf32>, %c_abs: memref<128xf32>,
      %cos_a: memref<128xf32>, %sin_a: memref<128xf32>,
      %cos_t: memref<128xf32>, %sin_t: memref<128xf32>,
      %score: memref<1xf32>) {
      %t_packed = aiex.dma_configure_task_for @packed_in {
        aie.dma_bd(%packed : memref<96xui8>, 0, 96, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 96, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_packed)
      %t_cnorm = aiex.dma_configure_task_for @cnorm_in {
        aie.dma_bd(%cnorm : memref<1xf32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cnorm)
      %t_c_mag = aiex.dma_configure_task_for @c_mag_in {
        aie.dma_bd(%c_mag : memref<128xf32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_c_mag)
      %t_c_abs = aiex.dma_configure_task_for @c_abs_in {
        aie.dma_bd(%c_abs : memref<128xf32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_c_abs)
      %t_cos_a = aiex.dma_configure_task_for @cos_a_in {
        aie.dma_bd(%cos_a : memref<128xf32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cos_a)
      %t_sin_a = aiex.dma_configure_task_for @sin_a_in {
        aie.dma_bd(%sin_a : memref<128xf32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_sin_a)
      %t_cos_t = aiex.dma_configure_task_for @cos_t_in {
        aie.dma_bd(%cos_t : memref<128xf32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_cos_t)
      %t_sin_t = aiex.dma_configure_task_for @sin_t_in {
        aie.dma_bd(%sin_t : memref<128xf32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_sin_t)
      %t_score = aiex.dma_configure_task_for @score_out {
        aie.dma_bd(%score : memref<1xf32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_score)
      aiex.dma_await_task(%t_score)
      aiex.dma_free_task(%t_packed)
      aiex.dma_free_task(%t_cnorm)
      aiex.dma_free_task(%t_c_mag)
      aiex.dma_free_task(%t_c_abs)
      aiex.dma_free_task(%t_cos_a)
      aiex.dma_free_task(%t_sin_a)
      aiex.dma_free_task(%t_cos_t)
      aiex.dma_free_task(%t_sin_t)
    }
  }
}
