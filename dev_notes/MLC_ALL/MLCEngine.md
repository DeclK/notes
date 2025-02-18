# MLCEngine

How `MLCEngine` in mlc-llm inference

## Concept

- `Request`

  可以看做一个存储结构，包含了以下部分

  - `String` id，输入的字符串
  - `Array<Data>` inputs，字符串 token ids + 图像数据
  - `GenerationConfig` generation_cfg，生成配置
  - `Object* rstate` rstate，一个指向 request state 的指针

- `RequestState`

  

- `request_stream_callback`

  这是一个 python function，在每次 step 最后进行调用，用于保存 decode 出的 outputs 结果

- `EngineState`

  It contains the requests and their states submitted to the Engine. 这个类里面保留的东西很多，基本上包含了 engine 运行中所需的信息

  - `vector<Request>` running_queue，包含了当前正在处理的 request 队列
  - `vector<Request>` waiting_queue，还没有处理的 request
  - `unordered_map<String, RequestState>` request_states，包含内容太多，目前不知道这么多信息拿来干嘛
  - `PrefixCache` prefix_cache，radix tree 保存的 cache
  - `EngineInternalIDManager` id_manager
  - `EngineMetrics` metrics，一些 runtime 指标总信息，包含 prefill/decode time sum, tokens sum...

## Layout

## Questions

- how to get metric
- what does engine state store?
- how engine manage workspace (both draft model and base model)
- how to use the ffi (foreign function interface) in tvm
- 这么看来 C++ 确实应该是一个比较好懂的面向对象的语言，由于其非常看重 type 的特性，我们可以直接查看该类包含了哪些 type/member 来理解该类的功能
- How does prefix kvcache being managed and used?
- How to build a nice event recorder in C++