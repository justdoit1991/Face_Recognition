# Face_Recognition
***
## Requirments
```
keras 2.2.0
tensorflow 1.9.0
opencv-python 4.1.2.30
```
## Process
```
1. Set model path
2. Use mtcnn find boundingbox
3. Extracting face from image based on boundingbox
4. Resize => 160*160
5. Use infer function calculation feature vector
6. Calculate cos distance between faces (Suggested threshold between 0.21 to 0.23)
```
***
## Example
```
```
