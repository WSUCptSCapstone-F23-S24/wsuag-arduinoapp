# Sprint 4 Report 3/2/2024

## What's New (User Facing)
 * New model iteration
 * Added NDVI Calculation for plots
 * Added growth curve for datasets


## Work Summary (Developer Facing)
During the latest sprint for the WSUAG Data Project, our team achieved significant advancements by enhancing our plot segmentation model with additional training on a diverse dataset of altered images, and integrating NDVI calculations to assess vegetation health. This led to the development of a feature for plotting NDVI values on growth curves, offering detailed insights into crop growth patterns over time. By incorporating these improvements, including the seamless integration of the advanced yolov8 model and streamlined data handling for efficient output generation, we've substantially elevated our project's capability to deliver precise agricultural insights, marking a pivotal step forward in our ongoing development efforts.

## Unfinished Work
We did not have any unfinished work in this sprint.



## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:
* [Image processing pipeline/growth curve](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/44)
* [Preliminary testing of version 2/3/4 of the model](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/45)
* [Improve and adjust dataset for training future iteration of models](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/46)
* [Create different iterations of segmentation model with different base yolov8 models](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/48)
* [Update project report and add sprint report and sprint video to readme](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/issues/47)

 ## Incomplete Issues/User Stories
* We completed all issues in this sprint


## Code Files for Review
Please review the following code files, which were actively developed during this sprint, for quality:
   * [plot_growth_curve.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/sprint4.2/src/plot_growth_curve_adjusted.py)
   * [predict.py](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/sprint4.2/src/predictnew.py)

 

## Retrospective Summary
Here's what went well:
   * Enhanced Model Accuracy: Training the segmentation model on an enriched dataset significantly improved its precision.
   * NDVI Implementation: Successfully integrating NDVI calculations into our pipeline has opened up new avenues for vegetation analysis.
   * Growth Curve Visualization: The introduction of growth curve plotting for NDVI values has significantly enriched our data presentation capabilities.
   * Streamlined Data Processing: The automation of data compilation and efficient export functionality have boosted our workflow productivity.
 
Here's what we'd like to improve:
   * Code Organization and Documentation: As our project's complexity increases, enhancing the organization and documentation of our repository will be crucial for maintainability.
   * Subsystem Integration: Ensuring more seamless interaction between different components of our system remains a priority for improving overall functionality and user experience.
  
Here are changes we plan to implement in the next sprint:
   * Advanced Data Analysis Features: We aim to introduce more sophisticated data analysis capabilities, leveraging deeper insights from the enhanced NDVI calculations and growth curve visualizations.

