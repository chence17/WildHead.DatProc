# Work.3DHeadGen.DatProc
## Refactor

### `filter`

`size_filter`: To filter image size.

`blur_filter`: To filter blurry image.

### `detector`

`head_detector`: To detect head.

### `visualize`

`vis_2d`: To visualize 2d images.

### `Cropper`

`front_view_cropper`: To crop front view images.

## Misc

Data Processing for PanoHead

## Notes

Thanks to PanoHead code for data processing. This repository is a fork of the original code with modification.

- `3DDFA_V2` git branch tag is `1b6c67601abffc1e9f248b291708aef0e43b55ae`.

## Configure Conda Environment

```bash
conda create -n datproc python=3.8
conda activate datproc
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install cudnn==8.2.1
pip install onnxruntime-gpu==1.14.1
pip install face-alignment
pip install opencv-python==4.5.4.58
pip install matplotlib imageio imageio-ffmpeg pyyaml tqdm argparse cython scikit-image scipy gradio
pip install dlib
```

## Demo

```bash
python dlib_kps.py
python recrop_images.py -i examples/test/data.pkl -j dataset.json -o examples/test/crop_samples/img
```
