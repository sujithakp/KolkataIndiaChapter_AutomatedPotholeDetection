### Model Training for Pothole Detection:

#### As a team, we decided with the YOLO and RCNN models as a sub teams to work on each model. Came up with the metric mAP to evaluate the model performance. We found the mAP achieved best with the YOLO11 so finalized best weighted custom model for the development & deployment phase
#### We then came up to specify damage severity in terms of percentage in the UI of the app by performing the training on instance segmentation
#### Worked on Data Collections, Data Pre-processing/Annotations, Model Development, Model Deployment and Documentation for SRS, DPR

> mAP I could achieve was 0.738 with the training I did using YOLO11 Medium with Instance Segmentation...

> Open Source tools used during the project are: Roboflow, Kaggle Notebook, Google Colab, Google Docs and Drive, VS Code, CodeSandBox.io, Presentation tool...

> I was task lead for Data Collection and Documentation phases - eventually worked on leading the Pre-processing and development by working on deployments tools and liasing with the team

> Collectively collected the datasets from Kaggle and Mendeley

> Further I distributed the set the images to the team for the annotations via Roboflow and integrated all the sets into one for next i.e., development...
 
> I worked on the model training using YOLO11 with Medium type on custom object detection and instance segmentation datasets

> Initially worked on the web app using Flask framwork and uploaded it to the online tool CodeSandBox for the collaborative development. Team member continued with the codebase from CodeSandBox that I shared with the team.
 
> Worked on the documentation for SRS and DPR and distributed with the collaborators to complete the parts of the work...

> Had an opportunity to explore on Microsoft Florence2 for auto annotations along with the Roboflow to customise for our project requirements.

> Collaborated, communicated and connected over online/screen sharings with the team on all the phases of the project...

> I proposed to do PWA for deploying to smart devices for both the variations of web app and mob app - eventually this will be in pipeline to develop/enhance the application

#### Below are screenshots of the artefacts I developed during the model training phase with the YOLO11:
![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/mAP_Values_SegDevelopment.png)

![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/results.png)

#### 1.1 Validation batch during the model development that shows the class name assigned during the annotation:
![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/val_batch0_labels.jpg)

#### 1.2 Prediction with Confidence threshold for the above validation batch:
![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/val_batch0_pred.jpg)

#### 2.1 Another Validation batch during model development:
![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/val_batch1_labels.jpg)

#### 2.2 Another Prediction for the above validation batch:
![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/val_batch1_pred.jpg)

#### 3.1 Prediction for the above validation batch:
![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/val_batch2_labels.jpg)

#### 3.2 Prediction for the above validation batch:
![home_page](https://github.com/OmdenaAI/KolkataIndiaChapter_AutomatedPotholeDetection/blob/main/ModelDevelopment_RN/InstanceSegmentation_Dev6/val_batch2_pred.jpg)
