# XGBoost

目前在优化 auto parallel 算法，希望使用 xgboost 来解决当前的决策问题

目前来看 xgboost 也可以看做一个回归模型，似乎并不能保住决策

## TVM AutoTune

- 初始化

  - cost model, model optimizer

  - trails, 记录下哪些组合是尝试过的

- XGBTuner

  - 通过多次 trials 找出最优的配置

  - 获得一个 `measure_batch` function，输入为 inputs，输出为所花费时间

  - 对于每一个 batch trial：

    - 通过 `self.next_batch` 获得一个 batch trials，相当于是 inputs，即多种可能的 configs

      ```python
      def next_batch(self, batch_size):
          """
          Specific usage:
          1. If the trial list is empty, randomly choose a config.
          2. If the trial list is not empty, choose a config from the trial list.
          3. If the trial list is empty and the tuner is doing the last 5% trials (e-greedy),
                choose randomly.
          """
      ```

    - 对 batch/inputs 中的每一个 input 获得一个 measured result

    - 保留最优的 result

    - 通过 `self.update(inputs, results)` 更新 xgboost。（只有累计了足够多的样本才会进行 fit）

      ```python 
      # cost model is xgboost
      # self.xs is inputs index
      # self.ys is measured inputs flops
      # self.plan_size is number of recall points
      self.cost_model.fit(self.xs, self.ys, self.plan_size)
      ```

    - 更新好 xgboost 过后使用 `self.model_optimizer` 采样生成 maximums，这些 maximums 就是下一次进行的 trials 的候选配置

      ```python 
      maximums = self.model_optimizer.find_maximums(
          self.cost_model, self.plan_size, self.visited
      )
      ```

      我们需要 xgboost 来估计 configs 的时间，找到一些比较好的 configs，它们的耗时在 xgboost 的估计下是较小的，这些 configs 就成为了 maximums

      你希望使用 cost model 来衡量你的 configs 而不是真实地去进行测试，因为在退火模拟的过程中会有很多次的 configs 尝试，这个开销是不可承受的

    - 循环往复，能够以较大的概率获得最优的配置
