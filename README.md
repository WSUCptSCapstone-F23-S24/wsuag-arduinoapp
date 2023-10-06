 # WSUAG-Arduino App

## Project summary

### One-sentence description of the project
Machine-Learning based image processing of agricultural data.

### Additional information about the project
The WSUAG-Arduino App project is a custom solution designed to process images and automatically extract crop digital data from the Washington State University AGIcam system. This pipeline utilizes TensorFlow to detect a reference file and apply radiometric correction to the images. A TensorFlow model is also applied to segment the crops by plot in order to ascertain their statuses over time.


## Installation

### Prerequisites

- [python 3.9](https://www.python.org/downloads/release/python-390/)
- [protobuf](https://github.com/protocolbuffers/protobuf/releases)
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- [tensorflow](https://github.com/tensorflow/models)

### Add-ons

See [conda_libraries.txt](https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp/blob/docs/conda_libraries.txt) for list of add ons used in tensorflow environment.

### Installation Steps

Steps:
Clone repo - git clone https://github.com/WSUCptSCapstone-F23-S24/wsuag-arduinoapp.git
Install protobuf: download release for your device type - https://github.com/protocolbuffers/protobuf/releases
Download anaconda -  https://www.anaconda.com/
In anaconda Prompt:
  conda create -n tf2 pip python=3.9
  conda activate tf2


## Functionality

For fucntionality specifically running our model on iamges, first open repo in a conda environemnt. 
Change the directory to the object_detection folder.
Add images to the crop_images folder that you would like to run the model on.
run the command: python .\detect_from_image.py -m ._inference_graph\saved_model\ -l .\labelmap.pbtxt -i .\test_images\crop_test
check the folder called ouput for the annotated pictures.


## Known Problems

A known issue in our project is cloning this repository. This is because our git repo uses "Large File Storage" (LFS) to store files larger than 100 mb which we do have.
When pulling the repo some files are not acccessbile so we have to run these commands in the terminal:
- git lfs fetch
- git lfs checkout


## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Additional Documentation

TODO: Provide links to additional documentation that may exist in the repo, e.g.,
  Sprint Report: 
  Project Description: 
  Conda Libraries: 

## License
MIT License

Copyright (c) 2023 wsuag-arduinoapp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
