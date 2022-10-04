# Sherlock
This repository contains experiments for different publications at the intersection of Computer Vision and Computer Security.

We are currently #1 on paperswithcode for malware detection: https://paperswithcode.com/dataset/malnet.

We are currently #1 on paperswithcode for malware detection from type labels: https://paperswithcode.com/dataset/malnet.

We are currently #1 on paperswithcode for malware detection from family labels: https://paperswithcode.com/dataset/malnet.

We are currently #1 on paperswithcode for malware type detection: https://paperswithcode.com/dataset/malnet.

We are currently #1 on paperswithcode for malware family detection: https://paperswithcode.com/dataset/malnet.

# Dataset [malnet dataset](http://malnet.cc.gatech.edu/image-data/)

## What is a binary image?

Binary images represent the bytecode of an executable as a 2D image (see figure below), and can be statically extracted from 
many types of software (e.g., EXE, PE, APK). We use the Android ecosystem due to its large market share, easy 
accessibility, and diversity of malicious software.

![Binary image](images/binary-image.png "Android APK binary image")

# Inference and regenerate results
Follow these steps to evaluate each model.
1. Download the dataset from [malnet dataset](http://malnet.cc.gatech.edu/image-data/) and prepare the data.
      
      * Download full-data-as-1GB or full-data-as-6GB and copy all the zip files to a folder.
      * To recombine file chunks after downloading, run:
        
         `cat malnet-image* | tar xzpvf -`
      * To create the required data files for binary, type and family training or evaluation, update the config file in data folder. Then run main.py as below.
        
        'groups' : ['family', 'binary','type'], # binary, 'type', 'family'
        
        'data_dir': Data folder path where the group should be created, 
        
        'image_dir': Image unzip folder path which is created from the previous step, 
        
        'dataset_type': what type of dataset to create from train, test and val, # all, train, test, val
        
        'symbolic': create symbolic links or copy images, # True, False
        

        `python data/main.py`

2. Download the checkpoints to your local folder

| Experiment | Classes (nb_classes) | Checkpoint (model_path) |
| ------  | ------ | ------ |
|Binary|2 | [binary.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=SentcupmZqR6GsNd7Cy5112822057&browser=true&filename=binary.pth) |
|Type|47| [type.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=TEvV9VPZeyFqrSIDWDmF112822061&browser=true&filename=type.pth) |
|Family|696| [family.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=b6V6auEggiwNgOnhDpEZ1128220607&browser=true&filename=family.pth) |

3. Execute the following commands to evaluate each experiment.

| Experiment | Command | 
| ------| ------|
|Binary|python regenerate_experiment_results.py --model_path model_path_to_Binary --nb_classes 2 --data_path data_path_to_Binary|
|Type|python regenerate_experiment_results.py --model_path model_path_to_Type --nb_classes 47 --data_path data_path_to_Type|
|Family|python regenerate_experiment_results.py --model_path model_path_to_Family --nb_classes 696 --data_path data_path_to_Family|

4. After the above step .csv files will be generated with results. Use those .csv files and run {binary/family/type}_classification_metrics_generation.py file to regenerate the results.

# Results
| Experiment | Classes | F1 | Precision | Recall | Checkpoint |
| ------ | ------ | ------ | ------ | ------ | ------ |
|Binary|2|.854 | .920| .810 | [binary.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=SentcupmZqR6GsNd7Cy5112822057&browser=true&filename=binary.pth) |
|Type|47| .497| .628| .447 | [type.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=TEvV9VPZeyFqrSIDWDmF112822061&browser=true&filename=type.pth) |
|Family|696| .491| .568| .461 | [family.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=aWPazKFmzZdRj2eXNJZP112822059&browser=true&filename=family.pth) |



