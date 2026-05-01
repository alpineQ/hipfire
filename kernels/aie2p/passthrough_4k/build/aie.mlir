module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in(%logical_shim_noc, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<1024xui8>> 
    aie.objectfifo @out(%logical_core, {%logical_shim_noc_0}, 2 : i32) : !aie.objectfifo<memref<1024xui8>> 
    func.func @passthrough_fn(%arg0: memref<1024xui8>, %arg1: memref<1024xui8>, %arg2: i32) {
      %c0 = arith.constant 0 : index
      %1 = arith.index_cast %arg2 : i32 to index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %1 step %c1 {
        %2 = memref.load %arg0[%arg3] : memref<1024xui8>
        memref.store %2, %arg1[%arg3] : memref<1024xui8>
      }
      return
    }
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %1 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %2 = aie.objectfifo.subview.access %1[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
        %3 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xui8>>
        %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<1024xui8>> -> memref<1024xui8>
        %c1024_i32 = arith.constant 1024 : i32
        func.call @passthrough_fn(%4, %2, %c1024_i32) : (memref<1024xui8>, memref<1024xui8>, i32) -> ()
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<4096xui8>, %arg1: memref<4096xui8>, %arg2: memref<4096xui8>) {
      %1 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xui8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<4096xui8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%1)
    }
  }
}

