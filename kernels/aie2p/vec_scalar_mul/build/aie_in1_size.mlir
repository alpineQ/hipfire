module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_1 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in(%logical_shim_noc, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo @infactor(%logical_shim_noc_0, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<1xi32>> 
    aie.objectfifo @out(%logical_core, {%logical_shim_noc_1}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    func.func private @vector_scalar_mul_vector(memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) attributes {link_with = "scale.o"}
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %1 = aie.objectfifo.acquire @infactor(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
        %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
        %c0_2 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c4 step %c1_3 {
          %3 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi16>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<1024xi16>> -> memref<1024xi16>
          %5 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xi16>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<1024xi16>> -> memref<1024xi16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @vector_scalar_mul_vector(%4, %6, %2, %c1024_i32) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @infactor(Consume, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<4096xi16>, %arg1: memref<1xi32>, %arg2: memref<4096xi16>) {
      %1 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%3)
      aiex.dma_await_task(%3)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
    }
  }
}

