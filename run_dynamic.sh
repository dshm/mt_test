python test_arcface_dynamic_batchsize.py --xpu musa --model /dsmtest/dsm_test/arcfaceresnet100-8_dynamic_batch.onnx

python test_ecapa_dynamic_batchsize.py --xpu musa --model  /models/ECAPA/voxceleb_ECAPA512.onnx

python test_hrnet_dynamic_batchsize.py --xpu musa --model /dsmtest/dsm_test/hrnet_w18_fp32_dynamic_batch.onnx

python test_retinaface_dynamic_batchsize.py --xpu musa --model /models/Retinaface/RetinaFace.onnx

python test_slowfast_dynamic_batchsize.py --xpu musa --model /dsmtest/slowfastenv/slowfast_dynamic.onnx

python test_yolov8_dynamic_batchsize.py --xpu musa --model /dsmtest/yoloenv/yolov8n_generated_dynamic.onnx

python test_fastspeech2_decoder_dynamic_batchsize.py --xpu musa --model /dsmtest/dsm_test/fastspeech2_csmsc_am_decoder_dynamic_batch.onnx

python test_fastspeech2_postnet_dynamic_batchsize.py --xpu musa --model /dsmtest/dsm_test/fastspeech2_csmsc_am_postnet_dynamic_batch.onnx


