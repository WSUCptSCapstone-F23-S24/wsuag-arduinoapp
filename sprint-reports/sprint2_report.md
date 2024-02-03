# Sprint 2 Report 11/5/2023

## What's New (User Facing)
 * Reference Panel Detection Algorithm: Our image detection model has been significantly enhanced, resulting in improved accuracy and performance. Additionally, we've introduced reference panel analysis to provide users with valuable insights into image data.
 * Radiometric Correction: Development of scripts for obtaining constant color values, allowing for precise radiometric correction. We've also implemented image correction and normalization code, ensuring that image data is consistent and accurate, even in varying lighting conditions.


## Work Summary (Developer Facing)
During Sprint 2, our team made significant progress towards completing the WSUAG Data Project. First, we significantly improved our Reference Panel Detection Model. It is now at the point where it can be reliably used for the next steps. We successfully developed and implemented the Reference Panel Detection Algorithm, which plays a crucial role in identifying the color reference panel in raw images of wheat plots, given the output of our Reference Panel Detection Model. We determined the appropriate constant color values required for our algorithms. Afterwards, we developed an algorithm to apply Radiometric Correction to the images, ensuring that our data is accurately adjusted for any deviations from the constant color values. Our team encountered some initial hurdles in refining the models but we were able to overcome them.

## Unfinished Work
We did not finish the plot segmentation models we did not anticipate the amount of time needed to create this model.


## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:

 * [Create testing and acceptance plan](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/31)
 * [Normalize images based on constant values](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/30)
 * [Determine constant values needed for proper radiometric correction](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/29)
 * [Improve precision of reference panel detection algorithm](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/28)
 * [Develop reference panel detection algorithm](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/27)
 * [Improve image detection model](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/26)
 * [Produce adjusted images using prior adjustment algo](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/9)
 * [Get image adjustment Algo from previous work and apply it using pixel information](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/7)
 * [Create Sprint 2 Report](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/23)


 ## Incomplete Issues/User Stories
 Here are links to issues we worked on but did not complete in this sprint:
 
 * [Develop initial plot segmentation model](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/32): We ran into a complication we did not anticipate the time required to develop this model.


## Code Files for Review
All modified files also copied into [src/scripts](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/tree/main/src) folder for clarity

Please review the following code files, which were actively developed during this sprint, for quality:
 * [detect_from_image.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/tf2.0/models/research/object_detection/detect_from_image.py)
 * [image_normalization.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/tf2.0/models/research/object_detection/image_normalization.py)
 * [python_image_processing.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/main/src/tf2.0/models/research/object_detection/python_image_processing.py)
 

## Retrospective Summary
Here's what went well:
  * Completely nearly all issues.
  * Made significant progress towards the end of our project.
 
Here's what we'd like to improve:
   * Time management.
   * Some confusion about what needed to be done.
  
Here are changes we plan to implement in the next sprint:
   * Add plot segmentation model.
   * Develop tests.
