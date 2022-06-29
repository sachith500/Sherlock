# Sherlock
This repository contains experiments for different publications at the intersection of Computer Vision and Computer Security.

# Inference and regenerate results
Follow these steps to evaluate each models.
1. Download the dataset from [malnet dataset](http://malnet.cc.gatech.edu/image-data/) and prepare the data.
      
      * Download full-data-as-1GB or full-data-as-6GB and copy all the zip files to a folder.
      * To recombine file chunks after downloading, run:
        
         `cat malnet-image* | tar xzpvf -`
      * To create the required data files for binary, type and family training or evaluation, please run the following command. 
    
         `python extract-images.py json_file_path`

2. Download the checkpoints to your local folder

| Experiment | Classes (nb_classes) | Checkpoint (model_path) |
| ------  | ------ | ------ |
|Binary|2 | [binary.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=SentcupmZqR6GsNd7Cy5112822057&browser=true&filename=binary.pth) |
|Type|47| [type.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=TEvV9VPZeyFqrSIDWDmF112822061&browser=true&filename=type.pth) |
|Family|696| [family.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=aWPazKFmzZdRj2eXNJZP112822059&browser=true&filename=family.pth) |

3. Execute the following commands to evaluate each experiment.

| Experiment | Command | 
| ------| ------|
|Binary|python regenerate_experiment_results.py --model_path model_path_to_Binary --nb_classes 2 --data_path data_path_to_Binary|
|Type|python regenerate_experiment_results.py --model_path model_path_to_Type --nb_classes 47 --data_path data_path_to_Type|
|Family|python regenerate_experiment_results.py --model_path model_path_to_Family --nb_classes 696 --data_path data_path_to_Family|

# Results
| Experiment | Classes | F1 | Precision | Recall | Checkpoint |
| ------ | ------ | ------ | ------ | ------ | ------ |
|Binary|2|.854 | **.920**| .810 | [binary.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=SentcupmZqR6GsNd7Cy5112822057&browser=true&filename=binary.pth) |
|Type|47| .876| .891| .862 | [type.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=TEvV9VPZeyFqrSIDWDmF112822061&browser=true&filename=type.pth) |
|Family|696| **.878**| .867| **.890** | [family.pth](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=aWPazKFmzZdRj2eXNJZP112822059&browser=true&filename=family.pth) |



