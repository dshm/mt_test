XPU_TYPE="cuda"     # Should be 'cuda' or 'musa'

echo "==arcface=="
python test_arcface_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/arcface/arcfaceresnet100-8_dynamic_batch.onnx
echo "==ECAPA=="
python test_ecapa_dynamic_batchsize.py --xpu $XPU_TYPE --model /models/ECAPA/voxceleb_ECAPA512.onnx
echo "==hrnet=="
python test_hrnet_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/hrnet/hrnet_w18_fp32_dynamic_batch.onnx
echo "==Retinaface=="
python test_retinaface_dynamic_batchsize.py --xpu $XPU_TYPE --model /models/Retinaface/RetinaFace.onnx
echo "==slowfast=="
python test_slowfast_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/slowfast/slowfast_dynamic.onnx
echo "==yolov8=="
python test_yolov8_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/yolov8/yolov8n_generated_dynamic.onnx
echo "==fastspeech2_decoder=="
python test_fastspeech2_decoder_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/fastspeech2/fastspeech2_csmsc_am_decoder_dynamic_batch.onnx
echo "==fastspeech2_postnet=="
python test_fastspeech2_postnet_dynamic_batchsize.py --xpu $XPU_TYPE --model /models_dynamicbatch/fastspeech2//dsmtest/dsm_test/fastspeech2_csmsc_am_postnet_dynamic_batch.onnx

cd ..
echo "==fastspeech2_encoder=="
python test_fastspeech2_encoder.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_encoder_infer.onnx
echo "==mb_melgan=="
python test_mb_melgan.py --xpu $XPU_TYPE --model /models/fastspeech2/mb_melgan_csmsc.onnx