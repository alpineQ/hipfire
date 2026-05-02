module {
  aie.device(npu2) {
    func.func private @zero_i32(memref<64x32xi32>) attributes {link_with = "mm_64x64x32.o"}
    func.func private @matmul_i8_i32(memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) attributes {link_with = "mm_64x64x32.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)
    aie.objectfifo @A_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @A_L3L2_1(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @A_L3L2_2(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @A_L3L2_3(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @A_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>> 
    aie.objectfifo @A_L2L1_1(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3, %tile_4_3, %tile_5_3, %tile_6_3, %tile_7_3}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>> 
    aie.objectfifo @A_L2L1_2(%mem_tile_4_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_4, %tile_1_4, %tile_2_4, %tile_3_4, %tile_4_4, %tile_5_4, %tile_6_4, %tile_7_4}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>> 
    aie.objectfifo @A_L2L1_3(%mem_tile_6_1 dimensionsToStream [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_5, %tile_1_5, %tile_2_5, %tile_3_5, %tile_4_5, %tile_5_5, %tile_6_5, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<64x64xi8>> 
    aie.objectfifo.link [@A_L3L2_0] -> [@A_L2L1_0]([] [])
    aie.objectfifo.link [@A_L3L2_1] -> [@A_L2L1_1]([] [])
    aie.objectfifo.link [@A_L3L2_2] -> [@A_L2L1_2]([] [])
    aie.objectfifo.link [@A_L3L2_3] -> [@A_L2L1_3]([] [])
    aie.objectfifo @B_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_0] -> [@B_L2L1_0]([] [])
    aie.objectfifo @B_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_1] -> [@B_L2L1_1]([] [])
    aie.objectfifo @B_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_2] -> [@B_L2L1_2]([] [])
    aie.objectfifo @B_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_3] -> [@B_L2L1_3]([] [])
    aie.objectfifo @B_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_4(%mem_tile_4_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_4_2, %tile_4_3, %tile_4_4, %tile_4_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_4] -> [@B_L2L1_4]([] [])
    aie.objectfifo @B_L3L2_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_5(%mem_tile_5_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_5_2, %tile_5_3, %tile_5_4, %tile_5_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_5] -> [@B_L2L1_5]([] [])
    aie.objectfifo @B_L3L2_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_6(%mem_tile_6_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_6_2, %tile_6_3, %tile_6_4, %tile_6_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_6] -> [@B_L2L1_6]([] [])
    aie.objectfifo @B_L3L2_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @B_L2L1_7(%mem_tile_7_1 dimensionsToStream [<size = 4, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_7_2, %tile_7_3, %tile_7_4, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<64x32xi8>> 
    aie.objectfifo.link [@B_L3L2_7] -> [@B_L2L1_7]([] [])
    aie.objectfifo @C_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_0_0, @C_L1L2_0_1, @C_L1L2_0_2, @C_L1L2_0_3] -> [@C_L2L3_0]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_1_0, @C_L1L2_1_1, @C_L1L2_1_2, @C_L1L2_1_3] -> [@C_L2L3_1]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_2_0, @C_L1L2_2_1, @C_L1L2_2_2, @C_L1L2_2_3] -> [@C_L2L3_2]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_3_0, @C_L1L2_3_1, @C_L1L2_3_2, @C_L1L2_3_3] -> [@C_L2L3_3]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_4_1(%tile_4_3, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_4_2(%tile_4_4, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_4_3(%tile_4_5, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_4(%mem_tile_4_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_4_0, @C_L1L2_4_1, @C_L1L2_4_2, @C_L1L2_4_3] -> [@C_L2L3_4]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_5_1(%tile_5_3, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_5_2(%tile_5_4, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_5_3(%tile_5_5, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_5(%mem_tile_5_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_5_0, @C_L1L2_5_1, @C_L1L2_5_2, @C_L1L2_5_3] -> [@C_L2L3_5]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_6_1(%tile_6_3, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_6_2(%tile_6_4, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_6_3(%tile_6_5, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_6(%mem_tile_6_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_6_0, @C_L1L2_6_1, @C_L1L2_6_2, @C_L1L2_6_3] -> [@C_L2L3_6]([0, 2048, 4096, 6144] [])
    aie.objectfifo @C_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_7_1(%tile_7_3, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_7_2(%tile_7_4, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L1L2_7_3(%tile_7_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @C_L2L3_7(%mem_tile_7_1 dimensionsToStream [<size = 8, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<8192xi32>> 
    aie.objectfifo.link [@C_L1L2_7_0, @C_L1L2_7_1, @C_L1L2_7_2, @C_L1L2_7_3] -> [@C_L2L3_7]([0, 2048, 4096, 6144] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_0(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_1(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_2(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_0_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_1_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_2_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_3_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_4(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_4_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_5(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_5_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_6(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_6_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c64 step %c1_1 {
          %0 = aie.objectfifo.acquire @C_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<64x32xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
          func.call @zero_i32(%1) : (memref<64x32xi32>) -> ()
          %c0_2 = arith.constant 0 : index
          %c32 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c32 step %c1_3 {
            %2 = aie.objectfifo.acquire @A_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<64x64xi8>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xi8>> -> memref<64x64xi8>
            %4 = aie.objectfifo.acquire @B_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi8>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xi8>> -> memref<64x32xi8>
            func.call @matmul_i8_i32(%3, %5, %1) : (memref<64x64xi8>, memref<64x32xi8>, memref<64x32xi32>) -> ()
            aie.objectfifo.release @A_L2L1_3(Consume, 1)
            aie.objectfifo.release @B_L2L1_7(Consume, 1)
          }
          aie.objectfifo.release @C_L1L2_7_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<4194304xi8>, %arg1: memref<4194304xi8>, %arg2: memref<4194304xi32>) {
      %0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 0, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 0, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 524288, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%4)
      %5 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 32, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%5)
      %6 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 131072, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%6)
      %7 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%7)
      %8 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 655360, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%8)
      %9 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%9)
      %10 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 64, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%10)
      %11 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 262144, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%11)
      %12 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%12)
      %13 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 786432, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%13)
      %14 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%14)
      %15 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 96, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%15)
      %16 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 393216, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%16)
      %17 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%17)
      %18 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 917504, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%18)
      %19 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%19)
      %20 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 128, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%20)
      %21 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%21)
      %22 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%22)
      %23 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 160, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%23)
      %24 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%24)
      %25 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%25)
      %26 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 192, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%26)
      %27 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%27)
      %28 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%28)
      %29 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 224, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%29)
      %30 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%30)
      %31 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%31)
      %32 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048576, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%32)
      %33 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1048576, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%33)
      %34 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%34)
      %35 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1572864, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%35)
      %36 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%36)
      %37 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048608, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%37)
      %38 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1179648, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%38)
      %39 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%39)
      %40 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1703936, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%40)
      %41 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%41)
      %42 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048640, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%42)
      %43 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1310720, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%43)
      %44 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%44)
      %45 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1835008, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%45)
      %46 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%46)
      %47 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048672, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%47)
      %48 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1441792, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%48)
      %49 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%49)
      %50 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 1966080, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%50)
      %51 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%51)
      %52 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048704, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%52)
      %53 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%53)
      %54 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%54)
      %55 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048736, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%55)
      %56 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%56)
      %57 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%57)
      %58 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048768, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%58)
      %59 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%59)
      %60 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%60)
      %61 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 1048800, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%61)
      %62 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%62)
      %63 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%63)
      aiex.dma_await_task(%0)
      aiex.dma_await_task(%5)
      aiex.dma_await_task(%10)
      aiex.dma_await_task(%15)
      aiex.dma_await_task(%20)
      aiex.dma_await_task(%23)
      aiex.dma_await_task(%26)
      aiex.dma_await_task(%29)
      aiex.dma_await_task(%32)
      aiex.dma_await_task(%37)
      aiex.dma_await_task(%42)
      aiex.dma_await_task(%47)
      aiex.dma_await_task(%52)
      aiex.dma_await_task(%55)
      aiex.dma_await_task(%58)
      aiex.dma_await_task(%61)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
      aiex.dma_free_task(%4)
      aiex.dma_free_task(%6)
      aiex.dma_free_task(%7)
      aiex.dma_free_task(%8)
      aiex.dma_free_task(%9)
      aiex.dma_free_task(%11)
      aiex.dma_free_task(%12)
      aiex.dma_free_task(%13)
      aiex.dma_free_task(%14)
      aiex.dma_free_task(%16)
      aiex.dma_free_task(%17)
      aiex.dma_free_task(%18)
      aiex.dma_free_task(%19)
      aiex.dma_free_task(%21)
      aiex.dma_free_task(%22)
      aiex.dma_free_task(%24)
      aiex.dma_free_task(%25)
      aiex.dma_free_task(%27)
      aiex.dma_free_task(%28)
      aiex.dma_free_task(%30)
      aiex.dma_free_task(%31)
      aiex.dma_free_task(%33)
      aiex.dma_free_task(%34)
      aiex.dma_free_task(%35)
      aiex.dma_free_task(%36)
      aiex.dma_free_task(%38)
      aiex.dma_free_task(%39)
      aiex.dma_free_task(%40)
      aiex.dma_free_task(%41)
      aiex.dma_free_task(%43)
      aiex.dma_free_task(%44)
      aiex.dma_free_task(%45)
      aiex.dma_free_task(%46)
      aiex.dma_free_task(%48)
      aiex.dma_free_task(%49)
      aiex.dma_free_task(%50)
      aiex.dma_free_task(%51)
      aiex.dma_free_task(%53)
      aiex.dma_free_task(%54)
      aiex.dma_free_task(%56)
      aiex.dma_free_task(%57)
      aiex.dma_free_task(%59)
      aiex.dma_free_task(%60)
      aiex.dma_free_task(%62)
      aiex.dma_free_task(%63)
      %64 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097152, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%64)
      %65 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 2097152, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%65)
      %66 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%66)
      %67 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 2621440, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%67)
      %68 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%68)
      %69 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097184, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%69)
      %70 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 2228224, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%70)
      %71 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%71)
      %72 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 2752512, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%72)
      %73 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%73)
      %74 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097216, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%74)
      %75 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 2359296, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%75)
      %76 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%76)
      %77 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 2883584, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%77)
      %78 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%78)
      %79 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097248, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%79)
      %80 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 2490368, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%80)
      %81 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%81)
      %82 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3014656, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%82)
      %83 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%83)
      %84 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097280, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%84)
      %85 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%85)
      %86 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%86)
      %87 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097312, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%87)
      %88 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%88)
      %89 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%89)
      %90 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097344, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%90)
      %91 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%91)
      %92 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%92)
      %93 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 2097376, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%93)
      %94 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%94)
      %95 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%95)
      aiex.dma_await_task(%64)
      aiex.dma_await_task(%69)
      aiex.dma_await_task(%74)
      aiex.dma_await_task(%79)
      aiex.dma_await_task(%84)
      aiex.dma_await_task(%87)
      aiex.dma_await_task(%90)
      aiex.dma_await_task(%93)
      aiex.dma_free_task(%65)
      aiex.dma_free_task(%66)
      aiex.dma_free_task(%67)
      aiex.dma_free_task(%68)
      aiex.dma_free_task(%70)
      aiex.dma_free_task(%71)
      aiex.dma_free_task(%72)
      aiex.dma_free_task(%73)
      aiex.dma_free_task(%75)
      aiex.dma_free_task(%76)
      aiex.dma_free_task(%77)
      aiex.dma_free_task(%78)
      aiex.dma_free_task(%80)
      aiex.dma_free_task(%81)
      aiex.dma_free_task(%82)
      aiex.dma_free_task(%83)
      aiex.dma_free_task(%85)
      aiex.dma_free_task(%86)
      aiex.dma_free_task(%88)
      aiex.dma_free_task(%89)
      aiex.dma_free_task(%91)
      aiex.dma_free_task(%92)
      aiex.dma_free_task(%94)
      aiex.dma_free_task(%95)
      %96 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145728, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%96)
      %97 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3145728, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%97)
      %98 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%98)
      %99 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3670016, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%99)
      %100 = aiex.dma_configure_task_for @B_L3L2_0 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 0, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%100)
      %101 = aiex.dma_configure_task_for @C_L2L3_1 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145760, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%101)
      %102 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3276800, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%102)
      %103 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%103)
      %104 = aiex.dma_configure_task_for @A_L3L2_1 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3801088, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%104)
      %105 = aiex.dma_configure_task_for @B_L3L2_1 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 65536, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%105)
      %106 = aiex.dma_configure_task_for @C_L2L3_2 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145792, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%106)
      %107 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3407872, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%107)
      %108 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%108)
      %109 = aiex.dma_configure_task_for @A_L3L2_2 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3932160, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%109)
      %110 = aiex.dma_configure_task_for @B_L3L2_2 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 131072, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%110)
      %111 = aiex.dma_configure_task_for @C_L2L3_3 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145824, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%111)
      %112 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 3538944, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%112)
      %113 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%113)
      %114 = aiex.dma_configure_task_for @A_L3L2_3 {
        aie.dma_bd(%arg0 : memref<4194304xi8>, 4063232, 131072, [<size = 8, stride = 0>, <size = 32, stride = 64>, <size = 64, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%114)
      %115 = aiex.dma_configure_task_for @B_L3L2_3 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 196608, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%115)
      %116 = aiex.dma_configure_task_for @C_L2L3_4 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145856, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%116)
      %117 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%117)
      %118 = aiex.dma_configure_task_for @B_L3L2_4 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 262144, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%118)
      %119 = aiex.dma_configure_task_for @C_L2L3_5 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145888, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%119)
      %120 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%120)
      %121 = aiex.dma_configure_task_for @B_L3L2_5 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 327680, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%121)
      %122 = aiex.dma_configure_task_for @C_L2L3_6 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145920, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%122)
      %123 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%123)
      %124 = aiex.dma_configure_task_for @B_L3L2_6 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 393216, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%124)
      %125 = aiex.dma_configure_task_for @C_L2L3_7 {
        aie.dma_bd(%arg2 : memref<4194304xi32>, 3145952, 65536, [<size = 2, stride = 524288>, <size = 8, stride = 256>, <size = 256, stride = 2048>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%125)
      %126 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%126)
      %127 = aiex.dma_configure_task_for @B_L3L2_7 {
        aie.dma_bd(%arg1 : memref<4194304xi8>, 458752, 65536, [<size = 8, stride = 524288>, <size = 32, stride = 64>, <size = 32, stride = 2048>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%127)
      aiex.dma_await_task(%96)
      aiex.dma_await_task(%101)
      aiex.dma_await_task(%106)
      aiex.dma_await_task(%111)
      aiex.dma_await_task(%116)
      aiex.dma_await_task(%119)
      aiex.dma_await_task(%122)
      aiex.dma_await_task(%125)
      aiex.dma_free_task(%97)
      aiex.dma_free_task(%98)
      aiex.dma_free_task(%99)
      aiex.dma_free_task(%100)
      aiex.dma_free_task(%102)
      aiex.dma_free_task(%103)
      aiex.dma_free_task(%104)
      aiex.dma_free_task(%105)
      aiex.dma_free_task(%107)
      aiex.dma_free_task(%108)
      aiex.dma_free_task(%109)
      aiex.dma_free_task(%110)
      aiex.dma_free_task(%112)
      aiex.dma_free_task(%113)
      aiex.dma_free_task(%114)
      aiex.dma_free_task(%115)
      aiex.dma_free_task(%117)
      aiex.dma_free_task(%118)
      aiex.dma_free_task(%120)
      aiex.dma_free_task(%121)
      aiex.dma_free_task(%123)
      aiex.dma_free_task(%124)
      aiex.dma_free_task(%126)
      aiex.dma_free_task(%127)
    }
  }
}

