# Visual Genome 
For Visual Genome, we follow the DATASET.MD by [Kaihua Tang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md)

# GQA

1. Images: GQA images need to be downloaded from [GQA website](https://cs.stanford.edu/people/dorarad/gqa/download.html) and need to be stored in  ./datasets/gqa/images/. 
2. Annotation and Metadata: image_data.json, GQA-SGG.h5, GQA-SGG-dicts.json, trainining and testing scene graphs can be downloaded from [here](https://rpi.box.com/s/sgkudnxgrmtgu7b6pwsuf3ucmw2h1ahl). 
Optionally, you can follow the following steps. 
    1. Download the training and validation scene graphs from [GQA website](https://cs.stanford.edu/people/dorarad/gqa/download.html). The validation scene graphs are considered as testing set.  
    2. Store the scene graphs in *./datasets/gqa/* folder. 
    3. Run create_dict.py located in *./datasets/gqa/* folder. This will create the dictionary file GQA-SGG-dicts.json. 
    4. Run create_image_data.py located in *./datasets/gqa/* folder. This will create the image_data.json file. 
    5. Run create_h5.py located in *./datasets/gqa/* folder. This will create the GQA-SGG.h5 file. 
