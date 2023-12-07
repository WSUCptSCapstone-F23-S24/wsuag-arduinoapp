# Sprint 3 Report 12/5/2023

## What's New (User Facing)
 * Plot Segmentation Model: We implemented a model for plot segmentation, allowing us to divide up the images by plot.


## Work Summary (Developer Facing)
During Sprint 3, our team made significant progress towards completing the WSUAG Data Project. We added the new plot segmentation model, which was created using mask-rcnn. We trained the model, and developed a code to pipe images into it as well as to pipe images out of it with the plots segmented.

## Unfinished Work
We did not have any unfinished work in this sprint.



## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:
* [Develop initial plot segmentation model](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/32)
* [Figure out what to do for images with multiple plates.](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/3)
* [Research migration to Azure](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/19)
* [Create Sprint3 Report](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/33)
* [Research Test Solutions](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/24)
* [Recreate model](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/20)

 ## Incomplete Issues/User Stories
* We completed all issues in this sprint


## Code Files for Review
Please review the following code files, which were actively developed during this sprint, for quality:
   * [detect_and_image_process.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/tf2.0/models/research/object_detection/detect_and_image_process.py)
   * [plot_training.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/mrcnn_segmenting/kangaroo-transfer-learning/plot_training.py)
   * [plot_prediction.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/mrcnn_segmenting/kangaroo-transfer-learning/plot_prediction.py)
   * [training.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/mrcnn_segmenting/kangaroo-transfer-learning/training.py)

 

## Retrospective Summary
Here's what went well:
   * We completed all of our tasks.
   * We are ahead of where we expected to be.
 
Here's what we'd like to improve:
   * Organization could be improved.
  
Here are changes we plan to implement in the next sprint:
   * Create testing and verify all code is working as expected.


Please review the following code files, which were actively developed during this sprint, for quality:
 * [detect_from_image.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/tf2.0/models/research/object_detection/detect_from_image.py)
 * [detect_and_image_process.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/tf2.0/models/research/object_detection/detect_and_image_process.py)
 * [plot_training.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/mrcnn_segmenting/kangaroo-transfer-learning/plot_training.py)
 * [plot_prediction.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/mrcnn_segmenting/kangaroo-transfer-learning/plot_prediction.py)

## Retrospective Summary
Here's what went well:
   * We were able to complete all issues made for this sprint
   * We were able to get some extra modifications done as well
     
Here's what we'd like to improve:
   * Origanizations of the repo
   * Integration of subsystems
     
Here are changes we plan to implement in the next sprint (next semester):
   * Clean and consice code that is easy to read