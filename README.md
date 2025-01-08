## Chess pieces recognition using YOLO models

### Codes can be found in the following branches:
- **accuracy** : Multi-camera based branch with accuracy checks 
- **accuracySingleCamera** : Multi-camera based branch with accuracy checks 

- The other branches are behind these final branches. However they show the process that how we developed the whole model.

- **Note**: PATHS inside the codebase should be changed accordingly if it is required to run

### How to run the whole project
- Go one of the two branches (accuracy or accuracySingleCamera)
- run ./start.sh in the project workspace
- if the overfitted model is not required add the argument "--no-accuracy"
- if no argument is given then automatically accuray model will start working.


### Demo Videos
Here are some demonstration videos of the project in action:


![Multi Camera YOLOv8 medium model segmentation based tracking](multiMediumNoAcc.mp4)

![Single Camera YOLOv8 medium model segmentation based tracking without ground-truth video](singleMediumNoACC.mp4)

![Single Camera YOLOv8 medium model segmentation based tracking with ground-truth video](singleMediumNoACC.mp4)
