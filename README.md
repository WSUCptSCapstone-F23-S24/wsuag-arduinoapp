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

TODO: Describe the installation process (making sure you mention `bundle install`).
Instructions need to be such that a user can just copy/paste the commands to get things set up and running. 


## Functionality

TODO: Write usage instructions. Structuring it as a walkthrough can help structure this section,
and showcase your features.


## Known Problems

TODO: Describe any known issues, bugs, odd behaviors or code smells. 
Provide steps to reproduce the problem and/or name a file or a function where the problem lives.


## Contributing

TODO: Leave the steps below if you want others to contribute to your project.

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Additional Documentation

TODO: Provide links to additional documentation that may exist in the repo, e.g.,
  * Sprint reports
  * User links

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
