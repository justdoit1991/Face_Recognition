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
```python 
# NOTE               out   kernel  stride  exp  bias   res    se     active id  
large_config_list = [[16,  (3, 3), (1, 1), 16,  False, False, False, 'RE',  0],
                     [24,  (3, 3), (2, 2), 64,  False, False, False, 'RE',  1],
                     [24,  (3, 3), (1, 1), 72,  False, True,  False, 'RE',  2],
                     [40,  (5, 5), (2, 2), 72,  False, False, True,  'RE',  3],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE',  4],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE',  5],
                     [80,  (3, 3), (2, 2), 240, False, False, False, 'HS',  6],
                     [80,  (3, 3), (1, 1), 200, False, True,  False, 'HS',  7],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS',  8],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS',  9],
                     [112, (3, 3), (1, 1), 480, False, False, True,  'HS', 10],
                     [112, (3, 3), (1, 1), 672, False, True,  True,  'HS', 11],
                     [160, (5, 5), (1, 1), 672, False, False, True,  'HS', 12],
                     [160, (5, 5), (2, 2), 672, False, True,  True,  'HS', 13],
                     [160, (5, 5), (1, 1), 960, False, True,  True,  'HS', 14]]
```
