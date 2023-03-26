# eo2sar

U-GAT-IT (Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation) is a recently proposed generative adversarial network (GAN) architecture designed to translate images between unpaired domains. The model leverages a cycle-consistency loss and an attention mechanism to generate high-quality images that are more closely aligned with the input images than traditional GANs.

The U-GAT-IT architecture consists of two encoder networks that encode the unpaired images into a shared latent space and two decoder networks that generate the translated images in each domain. A cycle-consistency loss is then used to ensure that the generated images are consistent with the original input images. This loss helps to improve the quality of the generated images and reduce the likelihood of mode collapse.

To further improve the quality of the generated images, U-GAT-IT employs an attention mechanism that uses a self-attention module to focus on relevant regions of the input images. The attention mechanism guides the generation process and helps to align the generated images with the input images. 

![generator](/assets/eo2sar%20generator.png)

Finally, a discriminator network is used to distinguish between the translated images and real images from the target domain. The generator networks are trained to deceive the discriminator into thinking that the generated images are real, while the discriminator is trained to correctly identify the real images.

![discriminator](/assets/eo2sar%20discriminator.png)

Converting EO images to SAR-like images demands that GANs accommodate the  significant geometric changes between the domains. One of the strengths of U-GAT-IT is its ability to handle image translations that require holistic and large shape changes. This is possible because of the attentional mechanism and adaptive normalization layers used in the model.

The attentional mechanism helps the model to focus on specific regions of an image that are important for the translation task. This allows U-GAT-IT to identify and manipulate key features of an image that need to be modified in order to achieve the desired translation. For example, in translating a EO image to a SAR-like image, the attention mechanism might focus on specific geometrical/image features, which are all important for differentiating between EO images and SAR images.

The adaptive normalization layers are another key component of U-GAT-IT. These layers help to normalize the style and appearance of the input and output images, so that they match better. In addition, the adaptive normalization layers allow U-GAT-IT to adapt to different styles and variations within a single domain, such as different variations of the EO or SAR images. This flexibility makes the model well-suited for handling large shape changes and variations in input EO images.

## Results

![results](/assets/eo2sar_result.png)

Row-wise caption of result:

1. Real optical EO image
2. Heat map of real optical EO image to fake optical EO image 
3. Fake optical EO image
4. Heat map of fake optical EO image to fake SAR image 
5. Fake SAR image
6. Heat map of fake optical EO image to fake SAR image to fake optical EO image
7. Fake optical EO image

## Usage

The pre-trained eo2sar model can be downloaded [here](https://drive.google.com/file/d/1oevqaRxLaWLBqWceGG2z67IUl1MpLq_N/view?usp=sharing). To run the pre-trained eo2sar model:

```bash
python app.py --device [cpu/cuda:N] --model_path [path to the pre-trained model] --img [path to the EO image]
```

The output SAR image is of size 256 px by 256 px. If you wish to attain a higher resolution, please train the model from scratch.