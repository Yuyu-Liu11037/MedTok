<h1 align="center">
  MedTok: Multimodal Medical Code Tokenizer
</h1>

快速debug（500个患者）：

```CUDA_VISIBLE_DEVICES=2,3,4,5 python MedTok_EHR.py --task lenofstay --debug```
自定义数量debug（100个患者）：
```CUDA_VISIBLE_DEVICES=2 python MedTok_EHR.py --task lenofstay --debug --debug_patients 100```