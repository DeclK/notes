# Learn TorchPilot 1 

## Concept

- exception hook

  `sys.excepthook`

  利用这个 hook 来获得更显眼的 error message

- MPI command world, dist manager

  感觉 api 有好几种，但是核心的几个概念就几个，到时候只需要对应不同的 api 来获得关键信息就行

  1. rank: global and local
  2. world size
  3. barrier: done by mpi
  4. broadcast: done by mpi
  5. abort: done by mpi

- logger 设计

  每一个文件都有自己单独的 logger

  这样