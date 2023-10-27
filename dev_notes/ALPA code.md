# ALPA code

只要一个重点：如何在 mesh 上测速算子

之后使用采样的方式来生成

## Layout

- `get_compute_cost`

  函数将计算所有的 `(stage, mesh)` pair 消耗的时间

  对于每一个 sub mesh

  - 获得一个 `sliced_virtual_meshes`，仅用于提供 num_devices

  - 使用 `generated_training_stages_2d` 生成 stages。这里的 stages 既包含了前向，也包含了反向传播

    实际上就是用两个循环，将每一个 Jaxpr 

    每一个 stage 是一个长度为 3 的 tuple: `(stage_indices, stage_config, autosharding_config)`

    其中 `stage_indices` 为一个长度为 4 的 tuple `(start, end, mesh_id, config_idx)`

  - 使用 `distributed_profile_on_mesh` profile 每一个 stages

    包含两个部分，一个是 `compile_all` 一个是 `profile_all`

    

  

  