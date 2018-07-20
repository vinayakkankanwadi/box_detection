## Box Detection

### 1. Program usage

python version: python 2, 3

package installation: cv2, numpy, matplotlib

```
python box_detection.py <input_dir> <output_dir>
```
The program will take all the images under the directory of <input_dir>, and the drawing results and .json results will be generated under the <output_dir>.

For example,
```
python box_detection.py ./input/ ./output/
```

**Note:** The json representation of a box is the co-ordinates of its four vertices, starting with the top-left point, with clock-wise order.

### 2. Algorithm explanation
The steps are as follows:

1. Grayscale the image


2. Binarization  
  * Finding threshold: sort the grayscale value by the number of pixels with that value; Pick the value as threshold which:
    1. making up less than 5% of total number of pixels
    2. 50% of total number of pixels less than its predecessor


2. Highlight "form" edges
  * A "form" is the largest combination of boxes whose edges are within one connected components.
  * Indentify all the forms by finding connected components with dominantly large size


3. Use morphological operators to prevent unclosed boxes edges

4. Flood fill the margin

5. Get the location of boxes by flood fill all the boxes

6. Sort the boxes
