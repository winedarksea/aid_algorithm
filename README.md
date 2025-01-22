# Autonomous Island Defense (AID) Algorithm
Image processing for sea/sky/obstacle segmentation and detection.

This algorithm was designed for theoretical ability to be implemented on a microcontroller with very limited RAM. 
Substantial optimization is still required.

### Algorithm Outline
- Phase 1:
    - Separable Gaussian filter for noise removal
    - Local window feature extraction
    - Segmentation prediction with decision tree
    - Postprocess:
        - skyline detection and sea/sky cleaning
        - smoke removal and dilation
        - Connected Component Labeling (CCL) (Hoshen-Kopelman)
    - Calculate beach arrival criteria for Phase 2
- Phase 2:
    - Option 1: heuristics with obstacles from CCL
    - Option 2: lightweight object detection NN, ie SSD MobileNet V2 (boat, truck 25% confidence)
    - Red cross filter
- Phase 3 (not yet implemented)
    - MOSSE correlation filter for tracking, gradual update on high confidence
    - region of interest calculation
    - small random offset for distributed arrival

#### To-Do
* Add more features as needed to help wakes and turbulent water be classified correctly as water. Add the entropy of the LBP histogram, the variance of the Sobel gradient magnitude, the local skewness of intensity, and the variance of the quantized Sobel gradient orientations.
* Add CCL to accuracy measure
* More fully parameterize inputs
* Get a LiteRT/TensorflowLite object detector working for Phase 2
* Implement local feature generalization in a single set of loops in C, importable here or in other C code
* Phase 3 implementation
* Expand unittests

![Algorithm example with no neural network usage](./img/output_no_NN.png)
