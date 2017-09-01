**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform feature extraction using Histogram of Oriented Gradients (HOG), apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Train a Linear Support Vector Machine (SVM) classifier on the extracted features.
* Implement a sliding-window technique with the trained classifier to detect vehicles in an image.
* Create a heatmap of recurring detections to reduce false positives.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[car]: ./examples/car.png
[non_car]: ./examples/non_car.png
[grayscale]: ./examples/grayscale.png
[hog_visualization]: ./examples/hog_visualization.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Training images

The code for this step is contained in the second code cell of the IPython notebook of the file called `vehicle_detection.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car][car]
![non car][non_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_visualization][hog_visualization]

#### 2. HOG Parameters

After exploring different color spaces and HOG parameters, following configuration was chosen which gave best test-set accuracy.
orientations=8, pixels_per_cell=(8,8), cells_per_block=(2,2)

#### 3. Training the classifier

LinearSVC is used as the classifier. The training process can be seen in the 8th cell of `Vehicle_Detection.ipynb`. The features are 
extracted and concatenated. The training images are scaled up to 0-255 before passed into the feature extractor.

The feature includes HOG features, spatial features & color histograms. 

### Sliding Window Search

#### 1. Detecting vehicles in unseen images

The code for detecting cars is in the cell 10.
Sliding window technique is used to search a portion of an image to predict whether or not a vehicle was present. In order to increase
efficiency, I reduced the search area by setting a region of interest which ignores the top half of the image & reduced the image by size of 1.5.
As the window slides along the search area, the classifier is used to predict whether or not a vehicle is present based on the features in that sample.

#### 2. Reduce false positives

To reduce false positives & to make bounding boxes consistent & smoother across frames, heatmaps are used of the positive detections reported by classifier.
An average of heatmaps over 15 past frames & used a threshold to remove false positives. I used scipy's label() function to identify "blobs" in the heatmap,
which correlated to vehicles in the image. The process_image() in cell 10 applies the threshold to the vehicle found. An example of an input image (left) and a heatmap 
applied to that image (right) is shown below:

#### 3. Visual display

Bounding boxes are displayed on the images around the detected cars using draw_bounding_boxes() for the given labels. The labels are nothing but blobs of heatmaps
mentioned above. By using an average of heatmaps over 15 frames of video, the result is a smooth & consistent bounding box around vehicles without any false positives
in the project video.


![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

