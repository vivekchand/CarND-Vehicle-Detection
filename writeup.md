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
[i0]: ./examples/pipeline/i0.png
[h0]: ./examples/pipeline/h0.png
[v0]: ./examples/pipeline/v0.png
[f0]: ./examples/pipeline/f0.png
[i1]: ./examples/pipeline/i0.png
[h1]: ./examples/pipeline/h0.png
[v1]: ./examples/pipeline/v0.png
[f1]: ./examples/pipeline/f0.png
[i2]: ./examples/pipeline/i0.png
[h2]: ./examples/pipeline/h0.png
[v2]: ./examples/pipeline/v0.png
[f2]: ./examples/pipeline/f0.png
[i3]: ./examples/pipeline/i0.png
[h3]: ./examples/pipeline/h0.png
[v3]: ./examples/pipeline/v0.png
[f3]: ./examples/pipeline/f0.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Training images

The code for this step is contained in the 2nd code cell of the IPython notebook of the file called `vehicle_detection.ipynb`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car][car]
![non car][non_car]


#### 2. HOG Parameters

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_visualization][hog_visualization]

The code for this step is in the 4th cell.

After exploring different color spaces and HOG parameters, following configuration was chosen which gave best test-set accuracy.
orientations=8, pixels_per_cell=(8,8), cells_per_block=(2,2)

#### 3. Training the classifier

LinearSVC is used as the classifier. The training process can be seen in the 8th cell of `Vehicle_Detection.ipynb`. The features are 
extracted and concatenated. The training images are scaled up to 0-255 before passed into the feature extractor.

The feature includes HOG features, spatial features & color histograms. 

### Sliding Window Search

The code for sliding window search is in the cell 11.

#### 1. Detecting vehicles in unseen images

Sliding window technique is used to search a portion of an image to predict whether or not a vehicle was present. In order to increase
efficiency, I reduced the search area by setting a region of interest which ignores the top half of the image & reduced the image by size of 1.5.
As the window slides along the search area, the classifier is used to predict whether or not a vehicle is present based on the features in that sample.

#### 2. Reduce false positives

To reduce false positives & to make bounding boxes consistent & smoother across frames, heatmaps are used of the positive detections reported by classifier.
An average of heatmaps over 15 past frames & used a threshold to remove false positives. I used scipy's label() function to identify "blobs" in the heatmap,
which correlated to vehicles in the image. The process_image() in cell 11 applies the threshold to the vehicle found. An example of an input image (left) and a heatmap 
applied to that image (right) is shown below:

#### 3. Visual display

Bounding boxes are displayed on the images around the detected cars using draw_bounding_boxes() for the given labels. The labels are nothing but blobs of heatmaps
mentioned above. By using an average of heatmaps over 15 frames of video, the result is a smooth & consistent bounding box around vehicles without any false positives
in the project video.

### Vehicle Detection Images

![Input Image][i0]
![Heatmap][h0]
![Vehicles found][v0]
![Final Image][f0]

![Input Image][i1]
![Heatmap][h1]
![Vehicles found][v1]
![Final Image][f1]

![Input Image][i2]
![Heatmap][h2]
![Vehicles found][v2]
![Final Image][f2]

![Input Image][i3]
![Heatmap][h3]
![Vehicles found][v3]
![Final Image][f3]

### Video Implementation

I have uploaded the video to youtube. It's available here [https://www.youtube.com/watch?v=1yXvzVo5t8Q](https://www.youtube.com/watch?v=1yXvzVo5t8Q)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Discussion

The pipelines works well on the project video, but is still not production ready. When there are multiple vehicles
the detection has to be improvised. Currently if a vehicle is occluded by another vehicle, the current pipeline
detects as one vehicle.

Also the pipeline needs to be improved to process the video in real time. 

And vehicles that are relatively far away are not detected. It should be improved to detect any car in the camera's view
without any overlapping just as humans do.
