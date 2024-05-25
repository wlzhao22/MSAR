# The implementation of **MSAR**
## Version info: pytorch 2.0.0, cuda 11.8, python 3.9.16

| Floders            | explanation                                               |
| ------------------ | --------------------------------------------------------- |
| Object_detection   | Run intance dectection on a single image or dataset       |
| Feature_extraction | Extact instance-level features or other types of features |
| IoU                | Calculate the IoU of the result                           |
| Exhaustiveness     | Calculate the Exhaustiveness of the result                |



Run on whole dataset:
```pythobn
python ./Object_detection/kcut_mbp_dataset.py
```

Run on one image:

```pythobn
python ./Object_detection/kcut_mbp_visualize.py
```

Extract instance-level feature by RoI or Masked-RoI:

```pythobn
python ./Feature_extraction/feature_extraction.py
```

Features whiten and Calculate mAP:

```pythobn
python ./Feature_extraction/feature_process.py
```



### Publisher 
**Yi-Bo Miao Xiamen University**
