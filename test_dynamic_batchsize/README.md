# mt_test_dynamic_batchsize

支持摩尔线程的模型在不同batchsize下的测试，并同cuda下的吞吐率进行对比

# 1. 创建 docker 容器
测试musa下数据时
```shell
docker run -it --privileged --shm-size=80G  --pid=host --network=host --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g ort-musa-cmcc:v0.1-cmcc /bin/bash
```
测试cuda下数据时，需要在启动docker时增加--gpus all参数
```shell
docker run -it --privileged --shm-size=80G  --pid=host --network=host --env MTHREADS_VISIBLE_DEVICES=all --gpus all  --shm-size=80g ort-musa-cmcc:v0.1-cmcc /bin/bash
```


# 2. 准备模型
为了对不同的batchsize进行测试，需要得到支持不同batchsize的模型文件。基于仓库mt_test中提到的模型文件进行修改，原仓库中提到的voxceleb_ECAPA512.onnx和RetinaFace.onnx本身就支持batchsize大于1的情况，因此我们直接使用原本的模型文件。fastspeech2_csmsc_am_encoder_infer.onnx和mb_melgan_csmsc.onnx仅能找到batchsize=1的实现，不做修改仍使用原仓库中提到的文件。对于剩下的.onnx文件，我们重新生成了支持batchsize大于1的模型文件，分别为arcfaceresnet100-8_dynamic_batch.onnx、hrnet_w18_fp32_dynamic_batch.onnx、slowfast_dynamic.onnx、yolov8n_generated_dynamic.onnx、fastspeech2_csmsc_am_decoder_dynamic_batch.onnx和fastspeech2_csmsc_am_postnet_dynamic_batch.onnx。将模型文件放入相应的目录如下：


```plain-text
/models/
├── ECAPA
│   └── voxceleb_ECAPA512.onnx
├── Retinaface
│   └── RetinaFace.onnx
└── fastspeech2
    ├── fastspeech2_csmsc_am_encoder_infer.onnx
    └── mb_melgan_csmsc.onnx

/models_dynamicbatch/
├── arcface
│   └── arcfaceresnet100-8_dynamic_batch.onnx
├── fastspeech2
│   ├── fastspeech2_csmsc_am_decoder_dynamic_batch.onnx
│   └── fastspeech2_csmsc_am_postnet_dynamic_batch.onnx
├── hrnet
│   └── hrnet_w18_fp32_dynamic_batch.onnx
├── slowfast
│   └── slowfast_dynamic.onnx
└── yolov8
    └── yolov8n_generated_dynamic.onnx

```

# 3. 安装onnxruntime

测试musa下数据时，安装onnxruntime-musa
```shell
pip uninstall onnxruntime
pip install onnxruntime-1.18.1-cp38-cp38-linux_x86_64.whl 
```

测试cuda下数据时，安装onnxruntime-gpu。测试本文档数据时安装的版本为onnxruntime-gpu 1.19.2
```shell
pip install onnxruntime-gpu
```

# 4. 运行模型对应的测试脚本

> XPU_TYPE should be either 'musa' or 'cuda'. 

```shell
python test_arcface_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/arcface/arcfaceresnet100-8_dynamic_batch.onnx
python test_ecapa_dynamic_batchsize.py --xpu $XPU_TYPE --model /models/ECAPA/voxceleb_ECAPA512.onnx
python test_hrnet_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/hrnet/hrnet_w18_fp32_dynamic_batch.onnx
python test_retinaface_dynamic_batchsize.py --xpu $XPU_TYPE --model /models/Retinaface/RetinaFace.onnx
python test_slowfast_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/slowfast/slowfast_dynamic.onnx
python test_yolov8_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/yolov8/yolov8n_generated_dynamic.onnx
python test_fastspeech2_decoder_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/fastspeech2/fastspeech2_csmsc_am_decoder_dynamic_batch.onnx
python test_fastspeech2_postnet_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/fastspeech2//dsmtest/dsm_test/fastspeech2_csmsc_am_postnet_dynamic_batch.onnx

cd ..
python test_fastspeech2_encoder.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_encoder_infer.onnx
python test_mb_melgan.py --xpu $XPU_TYPE --model /models/fastspeech2/mb_melgan_csmsc.onnx
```

注：当测试cuda下数据时，如果服务器上包含多个gpu，在执行命令前使用`CUDA_VISIBLE_DEVICES = x `指定具体的具体的gpu。本文档cuda部分数据测试时指定了`CUDA_VISIBLE_DEVICES = 2 ` ,即在第二个gpu上执行。

# 5. 测试结果

网络模型在不同batchsize下最大吞吐率测试结果如下(所测batchsize包括1, 4, 8, 16, 32, 64)：

| 网络名称                  | musa下最大吞吐率   | 最大吞吐率对应batchsize | 相同batchsize下cuda吞吐率 | cuda吞吐率10%| 10%要求 | 备注 |
|--------------------------|-----------|---|--|---|--|---|
| arcface             | 615.2239942    | 64 |2411.556839 | 241.1556839| 满足 | | 
| ecapa               | 2476.80796    | 64 | 8498.100716 | 849.8100716| 满足 | | 
| hrnet               | 353.5684903    | 64 |1590.507135 | 159.0507135| 满足 | | 
| retinaface          | 107.8791864    | 64 |329.0386501 |32.90386501| 满足| | 
| slowfast            | 18.8535993    | 64 |148.2239582 | 14.82239582| 满足| | 
| yolov8              | 281.5128541    | 64 | 430.2883224 |43.02883224| 满足| | 
| fastspeech2_decoder | 14560.71167    | 64 |10878.88729 |1087.888729| 满足| | 
| fastspeech2_postnet | 17251.30338    | 64 | 27732.94102 |2773.294102| 满足| | 
| fastspeech2_encoder | 2.80544255  | 1 | 11.06806862 |1.106806862| 满足| 仅支持batchsize=1的情况| 
| mb_melgan           | 170.9401709    | 1   |757.5757576 |75.75757576| 满足| 仅支持batchsize=1的情况| 

