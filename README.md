This is the repository of You Only Verify Once (YOVO) model, providing the model (and its weights) and the datasets.

YOVO is a lightweight CAPTCHA recognition model based on [YOLOv8n](https://github.com/ultralytics/ultralytics),
reformulating CAPTCHA recognition problem as a pure object detection task.

To get start, install Ultralytics and Timm in an environment:
```bash
pip install ultralytics
pip install timm
```

Before train or test, import the module YOVO used in your Ultralytics environment.
Add codes in `YOVO_Captcha/models/C2fFaster.py` to `your_env/ultralytics/nn/modules/block.py`.

Then registrate the module:

* Edit `your_env/ultralytics/nn/modules/block.py`
```python
__all__ = ("DFL", "HGBlock", "HGStem", "SPP", ...,
    "C2F_Faster",  # add
)
```

* Edit `your_env/ultralytics/nn/modules/__init__.py`
```python
from .block import (C1, C2, C2PSA, C3, ...,
    C2f_Faster,  # add
)
```
```python
__all__ = ("Conv", "Conv2", "LightConv", "RepConv", ...,
    "C2f_Faster",  # add
)
```

* Edit `your_env/ultralytics/nn/tasks.py`
```python
from ultralytics.nn.modules import (AIFI, C1, C2, C2PSA, ...,
    C2f_Faster,  # add
)
```
```python
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            ...,
            C2f_Faster,  # add
        }
    )
```

Now you could train the model with the architecture file, or use provided weights detect CAPTCHA images.
* YOVO's architecture .yaml file is provided in `YOVO_Captcha/models/YOVO.yaml`.
* Trained YOVO's weights on different datasets are also saved in `YOVO_Captcha/models/`.
* Datasets with images, labels and the .yaml file are provided in `YOVO_Captcha/datasets/`, unzip before use.
  * Dataset A: Synthetic fix-length 4 character CAPTCHA dataset (including 4 sub-parts).
  * Dataset B: Synthetic variable-length 3-6 character CAPTCHA dataset (including 9 sub-parts).
  * Upgraded Ganji: Real-world Ganji CAPTCHA Dataset upgraded from [here](https://github.com/SJTU-dxw/semi-supervised-for-captcha/tree/main).
    * Ganji_buchong: Annotations and images of unlabeled samples in the original Ganji Dataset.

The model can be trained or used with Ultralytics' official apis. Example codes:
* Train the model
```python
from ultralytics import YOLO

model = YOLO("YOVO_architecture_yaml_path")
dataset_yaml = "your_dataset_yaml_path"
output_dir = "your_output_path"

results = model.train(data=dataset_yaml, epochs=60, imgsz=224, batch=16, name="YOVO", project=output_dir, fliplr=0.0)
# fliplr=0.0 is used to disable horizontal flipping. You could also disable it at `your_env/ultralytics/cfg/default.yaml`, editing "fliplr: 0.5" to fliplr: 0"
```
* Detect images
```python
from ultralytics import YOLO

model = YOLO("your_weights_path")
img_path = "your_img_path"
output_dir = "your_output_path"

results = model.predict(img_path, save=True, save_txt=True, project=output_dir, name='predictions', exist_ok=True, conf=0.25, agnostic_nms=True, iou=0.5)
```
Please check [Ultralytics' official documents](https://docs.ultralytics.com/tasks/detect/#export) for more information.
