# MAX78000 YOLOv1 Demo



## Overview
The YOLOv1 Demo software demonstrates identification of a number of persons from their facial images using MAX78000 EVKit.

[comment]: <> (For this purpose, the CNN model generates a 512-length embedding for a given image, whose distance to whole embeddings stored for each subject is calculated. The image is identified as either one of these subjects or “Unknown”, depending on the embedding distances.)

[comment]: <> (The CNN model is trained with the VGGFace-2 dataset using MTCNN and FaceNet models for embedding generation.)

[comment]: <> (The code is auto-generated by the `ai8x-synthesis` tool and runs a known-answer test with a pre-defined input sample.)

## YOLOv1 Demo Software

### Building Firmware

Navigate directory where the YOLOv1 demo software is located and build the project:

```bash
$ cd Examples/MAX78000/CNN/yolov1_demo
$ make distclean
$ make
```

### Loading Firmware Image to Target

Connect a USB cable to CN1 (USB/PWR) and turn ON the power switch (SW1). Note the COM port (COM_PORT) of this connection from your system configuration.

Connect the PICO debug adapter to JH5 SWD header. 

Load firmware image using OpenOCD as described in the SDK documentation. **Make sure to remove PICO adapter once the firmware is loaded.**

### Host Application

#### Prerequisites
- Python 3.6.9 or higher (tested for 3.6.9, 3.7.7 and 3.8.1)
- NumPy (>1.18)
- Scipy (1.4)
- PyQt5 (5.9.2)
- OpenCv (>3.4)
- PySerial (>3.4)
- MatplotLib (>3.2)
- PyTorch (>1.3.1)
- TorchVision (>0.5.0)

If an `ai8x-synthesis` virtual environment does not already exist, follow the instructions in the main repository before continuing.

Add the additional packages using:

```shell
(ai8x-synthesis) $ pip3 install -r requirements-faceid.txt
```

#### Running the Host Application

Navigate to directory `demo` and run the `run_demo.py` script:

```bash
$ cd Examples/MAX78000/CNN/yolov1_demo/demo
$ python run_demo.py -c <COM_PORT>
```

`<COM_PORT>` is the Windows serial port identifier (e.g., `COM1`) or a Linux or macOS device (e.g., `/dev/tty.usbserial-D308XSRX`)

When the demo window is open, it is possible to load images from disk or capture images from the PC web cam. Currently, the app database includes images for five female and five male celebrities.

[comment]: <> (## CNN Model Design)

[comment]: <> (### Problem Definition)

[comment]: <> (* Identify people from three-channel &#40;RGB&#41; frontal facial images, i.e., portraits.)

[comment]: <> (* A small amount of rotation should be acceptable for robustness.)

[comment]: <> (### Approach)

[comment]: <> (The main approach in the literature is composed of three steps:)

[comment]: <> (- Face Extraction: Detection of the faces in the image and extract a rectangular sub-image that contains only a face.)

[comment]: <> (- Face Alignment: The rotation angles &#40;in 3D&#41; of the face in the image is found to compensate its effect by affine transformation.)

[comment]: <> (- Face Identification: The extracted sub-image is used to identify the person.)

[comment]: <> (In this project, the aim is to run all those steps in a single MAX78000 device, so the approach is to identify individual faces from uncropped portraits, each image containing a single face only.)

[comment]: <> (Then, the embeddings &#40;Face ID&#41; are created by a FaceNet [2] model as seen below, and these embeddings as the target. There is no need to deal with center loss, triplet loss etc, since those are assumed to be covered by the FaceNet model. The loss used in the model development will be Mean Square Error &#40;MSE&#41; between the target and predicted embeddings.)

[comment]: <> (### CNN Model)

[comment]: <> (The CNN model synthesized for MAX78000 is a 9-layer sequential model as shown below. It takes a 160×120 RGB image from the input and returns an embedding of length 512 corresponding to the image.)

[comment]: <> (```python)

[comment]: <> (class AI85FaceIDNet&#40;nn.Module&#41;:)

[comment]: <> (    """)

[comment]: <> (    Simple FaceNet Model)

[comment]: <> (    """)

[comment]: <> (    def __init__&#40;)

[comment]: <> (            self,)

[comment]: <> (            num_classes=None, )

[comment]: <> (            num_channels=3,)

[comment]: <> (            dimensions=&#40;160, 120&#41;,)

[comment]: <> (            bias=True,)

[comment]: <> (    &#41;:)

[comment]: <> (        super&#40;AI85FaceIDNet, self&#41;.__init__&#40;&#41;)

[comment]: <> (        self.conv1 = ai8x.FusedConv2dReLU&#40;num_channels, 16, 3, padding=1,)

[comment]: <> (                                          bias=False&#41;)

[comment]: <> (        self.conv2 = ai8x.FusedMaxPoolConv2dReLU&#40;16, 32, 3, pool_size=2, pool_stride=2,)

[comment]: <> (                                                 padding=1, bias=False&#41;)

[comment]: <> (        self.conv3 = ai8x.FusedMaxPoolConv2dReLU&#40;32, 32, 3, pool_size=2, pool_stride=2,)

[comment]: <> (                                                 padding=1, bias=bias&#41;)

[comment]: <> (        self.conv4 = ai8x.FusedMaxPoolConv2dReLU&#40;32, 64, 3, pool_size=2, pool_stride=2,)

[comment]: <> (                                                 padding=1, bias=bias&#41;)

[comment]: <> (        self.conv5 = ai8x.FusedMaxPoolConv2dReLU&#40;64, 64, 3, pool_size=2, pool_stride=2,)

[comment]: <> (                                                 padding=1, bias=bias&#41;)

[comment]: <> (        self.conv6 = ai8x.FusedConv2dReLU&#40;64, 64, 3, padding=1, bias=bias&#41;)

[comment]: <> (        self.conv7 = ai8x.FusedConv2dReLU&#40;64, 64, 3, padding=1, bias=bias&#41;)

[comment]: <> (        self.conv8 = ai8x.FusedMaxPoolConv2d&#40;64, 512, 1, pool_size=2, pool_stride=2,)

[comment]: <> (                                             padding=0, bias=False&#41;)

[comment]: <> (        self.avgpool = ai8x.AvgPool2d&#40;&#40;5, 3&#41;&#41;)

[comment]: <> (    def forward&#40;self, x&#41;:)

[comment]: <> (        x = self.conv1&#40;x&#41;)

[comment]: <> (        x = self.conv2&#40;x&#41;)

[comment]: <> (        x = self.conv3&#40;x&#41;)

[comment]: <> (        x = self.conv4&#40;x&#41;)

[comment]: <> (        x = self.conv5&#40;x&#41;)

[comment]: <> (        x = self.conv6&#40;x&#41;)

[comment]: <> (        x = self.conv7&#40;x&#41;)

[comment]: <> (        x = self.conv8&#40;x&#41;)

[comment]: <> (        x = self.avgpool&#40;x&#41;)

[comment]: <> (        return x)

[comment]: <> (```)


[comment]: <> (## References)

[comment]: <> ([1] MTCNN: https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)

[comment]: <> ([2] FaceNet: https://arxiv.org/pdf/1503.03832.pdf)
