# Data Augmentation Transforms

This project supports various data augmentation techniques designed to enhance your dataset. Below is an overview of each available transform.

## Resize

**Description:**
The `resize` transform adjusts the dimensions of an image to a specified size. This is useful for standardizing the input sizes of your dataset, which is often a requirement for deep learning models.

**Parameters:**  

- `target_height` : Specifies the target height of the output image. The image will be resized to this height.  
- `target_width` : Specifies the target width of the output image. The image will be resized to this width.  
- `keep_aspect_ratio` : If set to True, the transform will preserve the aspect ratio of the original image. The image will be resized to fit within the target dimensions while maintaining its aspect ratio.  
- `keep_aspect_ratio_mode` : Defines the behavior when keep_aspect_ratio is set to True.  
> - `not_larger`: The image is resized so that neither dimension exceeds the specified target dimensions. This may result in an image smaller than the target size in one or both dimensions.  
> - `not_smaller`: The image is resized so that neither dimension is smaller than the specified target dimensions. This may result in an image larger than the target size in one or both dimensions.  


- `letter_pad` : If set to True, padding will be added to the image to match the exact target dimensions after resizing. This is useful when maintaining the aspect ratio would otherwise result in an image smaller than the target dimensions.  
- `letter_pad_value` : Specifies the padding value(s) to be used when letter_pad is enabled. Can be a single integer value (e.g., 0 for black padding) or a tuple of three values for padding in RGB channels.

**Example Usage:**
```python
resize_cfg = ResizeConfig(
    target_height=640,
    target_width=640,
    keep_aspect_ratio=True,
    keep_aspect_ratio_mode="not_larger",
)
```


## RandomSingleDirectionFlip

**Description:**
The `RandomSingleDirectionFlip` transform randomly flips an image along a specified direction. This can be useful in data augmentation to introduce variation in the dataset, helping to improve the robustness of deep learning models.

**Parameters:**  

- `probability` (float) : Specifies the probability of applying the flip. This should be a value between 0 and 1, where 1 means the flip will always be applied and 0 means it will never be applied.  
- `flip_direction` (str): Defines the direction in which the image will be flipped.  
> - `'horizontal'`: The image will be flipped horizontally (left to right).  
> - `'vertical'`: The image will be flipped vertically (top to bottom).  


**Example Usage:**
```python
flip_cfg = RandomSingleDirectionFlipConfig(probability=0.5, 
                                           flip_direction="horizontal")
```


## RandomHSVAug

**Description:**
The `RandomHSVAug` transform randomly adjusts the hue, saturation, and value (brightness) of an image in the HSV color space. This is useful for augmenting the color properties of images, making the model more robust to variations in lighting and color.

**Parameters:**  

- `hue_delta` (float) : Specifies the maximum change to the hue component. The hue adjustment will be randomly selected within the range `[-hue_delta, hue_delta]`. Typical values are small fractions, as hue shifts can have a significant visual impact.  
- `saturation_delta` (float) : Specifies the maximum change to the saturation component. The saturation adjustment will be randomly selected within the range `[-saturation_delta, saturation_delta]`. Higher values introduce more vivid or more muted colors.  
- `value_delta` (float) : Specifies the maximum change to the value (brightness) component. The value adjustment will be randomly selected within the range `[-value_delta, value_delta]`. This parameter affects the brightness of the image.  
- `probability` (float): Defines the probability of applying the HSV augmentation. This should be a value between 0 and 1, where 1 means the augmentation will always be applied and 0 means it will never be applied.


**Example Usage:**
```python
hsv_cfg = RandomHSVConfig(
    hue_delta=0.015, saturation_delta=0.7, value_delta=0.4, probability=0.5
)
```

## Blur

**Description:**
The `Blur` transform applies a blurring effect to an image with a specified probability. Blurring can help simulate scenarios like out-of-focus images, adding robustness to the model by introducing a variety of image qualities.

**Parameters:**  

- `blur_limit` (int) : Defines the maximum kernel size for the blurring operation. The kernel size will be randomly selected between 3 and blur_limit, ensuring that the kernel size is always odd.  

- `probability` (float): Specifies the probability of applying the blur effect. This should be a value between 0 and 1, where 1 means the blur will always be applied, and 0 means it will never be applied.


**Example Usage:**
```python
blur_cfg = BlurConfig(blur_limit=7, probability=0.01)
```

## MedianBlur

**Description:**
The `MedianBlur` transform applies a median blurring effect to an image. Median blurring is particularly effective in reducing noise while preserving edges, making it a useful augmentation technique for various image processing tasks.

**Parameters:**  

- `blur_limit` (int) : Defines the maximum kernel size for the median blur operation. The kernel size will be randomly selected between 3 and blur_limit, ensuring that the kernel size is always odd.  

- `probability` (float): Specifies the probability of applying the median blur effect. This should be a value between 0 and 1, where 1 means the blur will always be applied, and 0 means it will never be applied.


**Example Usage:**
```python
blur_cfg = MedianBlurConfig(blur_limit=7, probability=0.01)
```


## CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Description:**
The `CLAHE` transform applies Contrast Limited Adaptive Histogram Equalization to an image. This technique is useful for enhancing the contrast in images, especially in regions with varying lighting conditions. It prevents over-amplification of noise by limiting the contrast in homogeneous areas. The implementation of this transform is aligned with the functionality provided by the Albumentations library.  

**Parameters:**  

- `clip_limit` (float) : Sets the threshold for contrast limiting. Higher values allow more contrast, while lower values restrict it to prevent noise amplification.

- `tile_grid_size` (tuple of int) : Defines the number of tiles in the grid over which the histogram equalization is applied. This is specified as a tuple (grid_width, grid_height). Smaller tiles result in better local contrast enhancement but may introduce noise if the tile size is too small.


- `probability` (float): Specifies the probability of applying the CLAHE effect. This should be a value between 0 and 1, where 1 means CLAHE will always be applied, and 0 means it will never be applied.


**Example Usage:**
```python
clahe_cfg = ClaheConfig(clip_limit=4, tile_grid_size=(8, 8), probability=0.01)
```


## Mosaic

**Description:**
The `Mosaic` transform combines four images into a single mosaic image, arranged in a grid format. This augmentation technique is particularly useful for object detection tasks, as it allows models to see objects in varied contexts and scales. The implementation is aligned with the functionality provided by the mmyolo library.

**Parameters:**  

- `mosaic_height` (int) : Image height after mosaic pipeline of single image.  
- `mosaic_height` (int) : Image height after mosaic pipeline of single image.  
- `fill_val` (float): The value used to fill in the empty spaces in the mosaic where images do not overlap. This is often a pixel value like 114.0 (commonly used for mean pixel values in datasets).
- `center_ratio_range` (tuple of float) : Center ratio range of mosaic output. Defaults to (0.5, 1.5).
- `probability` (float): Specifies the probability of applying the mosaic transformation. This should be a value between 0 and 1, where 1 means the transformation will always be applied.


**Example Usage:**
```python
clahe_cfg = ClaheConfig(clip_limit=4, tile_grid_size=(8, 8), probability=0.01)
```