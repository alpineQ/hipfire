// asym3_dequant_256/aie.mlir — single-head asym3 dequant for AIE-2P.
//
// Hand-written MLIR (no IRON Python). Three ObjectFifos shuttle data
// from shim DMA to one core tile and back:
//
//   packed_in:  shim → core, 96 bytes (256 × 3-bit indices packed)
//   cnorm_in:   shim → core,  4 bytes (one f32 scaling factor)
//   bf16_out:   core → shim, 512 bytes (256 × bf16)
//
// The core repeatedly acquires one element from each input FIFO + one
// from the output FIFO, calls the external C++ kernel
// `asym3_dequant_256`, and releases. The runtime_sequence sets up
// three host-side DMAs.
//
// Build: see Makefile in this directory. Tools used: aiecc (native
// C++), clang (Peano AIE-2P backend). No Python in the pipeline.

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

    // External C++ kernel — body lives in asym3_dequant_kernel.cc and
    // gets linked in by aiecc when it builds the core ELF.
    func.func private @asym3_dequant_256(memref<96xui8>, memref<1xf32>, memref<256xbf16>)

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

        %o_view = aie.objectfifo.acquire @bf16_out(Produce, 1)
            : !aie.objectfifosubview<memref<256xbf16>>
        %o = aie.objectfifo.subview.access %o_view[0]
            : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>

        func.call @asym3_dequant_256(%p, %c, %o)
            : (memref<96xui8>, memref<1xf32>, memref<256xbf16>) -> ()

        aie.objectfifo.release @packed_in(Consume, 1)
        aie.objectfifo.release @cnorm_in(Consume, 1)
        aie.objectfifo.release @bf16_out(Produce, 1)
      }
      aie.end
    } { link_with = "asym3_dequant_kernel.o" }

    // Host-side DMA orchestration. Three input/output buffers sized
    // to match a single dequant call (one head, one position).
    aie.runtime_sequence(%packed: memref<96xui8>, %cnorm: memref<1xf32>, %out: memref<256xbf16>) {
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

      %t_out = aiex.dma_configure_task_for @bf16_out {
        aie.dma_bd(%out : memref<256xbf16>, 0, 256,
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
