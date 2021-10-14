# Image colorization using GANs

The task of this project is to use GANs to colorize grey-scale images, and compare the results of the CycleGAN model, Pix2Pix model and U-Net generator with no discriminator model.

*NOTE* this is a Google Colab Notebook

# Introduction

Image colorization has long been recognized as highly laborious and tedious work. Despite advances in the automation of the process, a considerable amount of manual effort was still required before the emergence of deep neural networks. 

The task is a challenging problem due to the varying image conditions that need to be dealt with via a single model. The problem is also severely ill-posed as two out of three image dimensions are missing. Even though semantics of the scene may be helpful to achieve the desired results (the sky is blue and the grass is usually green), such semantic priors are highly uncommon for objects such as t-shirts, cars, tables and many other items.

Even though the real colors of the image are gone the moment it’s transferred to a grey-scale image, the fact that colors are predictable (up to an extent) makes this problem ideal for Machine Learning.

# Lab color space

Lab color space is a 3-channels representation of images, just like RGB, where the L stands for ‘lightness’ of the pixel and ab are the values of the color of the pixel. After reviewing a few papers, I noticed that they all first convert images to Lab color space and then predict the image colors. 

The reason for utilizing Lab color space is because Lab contains dedicated channel to depict the brightness (grey level) of the pixel and the color information is fully encoded in the remaining two channels. As a result, this prevents any sudden variations in both color and brightness through small perturbations in intensity values that are experienced through RGB.

# Dataset

The dataset I initially used was places365, a huge dataset of different places such as: shops, bedrooms, outdoors, bars and more. 
The problem with this dataset was the huge number of different objects and their non-consistent colors such as cars, clothes, etc. 
I had real hard time achieving any visible results, so I went with a different dataset with easier and more consistent semantics to color – Landscape dataset (can be found here Landscape Pictures | Kaggle).
The new Landscape dataset contains 4319 images of different landscape views such as mountains, beaches and fields, and it made the task of learning the corresponding colors of different objects easier.

<p float="center" align="center">
  <img src="https://user-images.githubusercontent.com/66253761/137330568-4580adb8-8c5a-4f78-9fcb-cb9422400fe2.png" alt="alt text">
</p>

# GANs

Or Generative Adversarial Networks, aim to model the natural image distribution by forcing the generated sample to be indistinguishable from natural images. GANs enable a wide variety of applications, such as image generation, image manipulation, object detection and more.

The GAN architecture is comprised of two models: a generator and a discriminator. The generator network generates candidates while the discriminator network evaluates them.
The discriminator is trained directly on real and generated images and is responsible for classifying images as real or fake (generated), whereas the generator is not trained directly and instead is trained via the discriminator model.
The two models compete in a two-player game, where simultaneous improvements are made to both generator and discriminator models.

# Pix2PIx

Is based on conditional GAN architecture. cGANs are trained on paired set of images or scenes from two domains to be used for image translation. 
The architecture of pix2pix consists of a generator and a discriminator. The generator is a encoder-decoder network (or U-Net with skip connections) while the discriminator is usually a patch-based discriminator, which penalizes at scale of patches. 

<p float="center" align="center">
  <img src="https://user-images.githubusercontent.com/66253761/137331361-f1807257-6179-4a68-8155-968ecd821378.png" alt="alt text">
</p>

# CycleGAN

For many image-to-image translation tasks, paired training data will not be available, just like in the zebra-horse example we learned about in class. 
CycleGAN is an approach to learn the mapping of input and output images using unpaired dataset. 
This model is an extension of Pix2Pix architecture, and it involves simultaneous training of two generator models and two discriminator models.

<p float="center" align="center">
  <img src="https://user-images.githubusercontent.com/66253761/137331057-8651f528-8756-4796-bec2-0e26eef5d83f.png" alt="alt text">
</p>

# Architectures

Generator:

The generator is based on the U-Net architecture we learned about, with skip connections, according to the CycleGAN/Pix2Pix paper, starting from images of size 256x256, down all the way to 1x1 and back up to 256x256. 
Input and output channels depend on the task, but for grey-to-color the input channel is 1 (l) and output is 2 (ab).

<p float="center" align="center">
  <img src="https://user-images.githubusercontent.com/66253761/137330475-0e029404-1c83-46a9-8f5d-e7ce64b05860.png" alt="alt text">
</p>

Discriminator:

The discriminator is a patch-based discriminator architecture that holds high-frequency structural information of the generator’s output by focusing on the local patches rather than the entire image. This gives a more localized real/fake decision rather than a binary decision for the whole image.

<p float="center" align="center">
  <img src="https://user-images.githubusercontent.com/66253761/137330461-172a7a48-e7c7-4a25-96ec-3673b3b76f0e.png" alt="alt text" width="300" height="200">
</p>

# Loss

I attempted a few combinations of loss functions, and the ones I chose are:

Gan Critic – is basically MSE loss between the discriminator output map and a map of zeros or ones, depending on whether the image was real or fake.

L1 Critic – is the L1 loss between the generated color channels and the real ‘ab’ color channels.

Cycle Loss – for the CycleGAN I also used a cycle consistency loss to try and minimize the distance of A->B->A’ image. Also L1 loss between the generated A’ and original A.

<p float="center" align="center">
  <img src="https://user-images.githubusercontent.com/66253761/137332162-51f5a9a4-4355-462c-b5e7-86e1527d2ff1.png" alt="alt text">
</p>

# Results

<p float="center" align="center">
  <img src="https://user-images.githubusercontent.com/66253761/137333159-bfa213d1-0985-4716-b317-419cc7ce9b0d.png" alt="alt text">
  <img src="https://user-images.githubusercontent.com/66253761/137333388-97ac387f-5156-4101-95bd-ecca9dcb55b7.png" alt="alt text">
  <img src="https://user-images.githubusercontent.com/66253761/137333510-5f94d0f2-7aba-44d0-a7b2-df84c7d97aee.png" alt="alt text">
  <img src="https://user-images.githubusercontent.com/66253761/137333611-ad68e789-f0ee-4e37-b2c3-12bbf34d9b11.png" alt="alt text">
  <img src="https://user-images.githubusercontent.com/66253761/137333700-50da6841-af3a-4f19-a4f2-e074dd2dd256.png" alt="alt text">
  <img src="https://user-images.githubusercontent.com/66253761/137333814-dad10b50-b0f0-458e-abd5-4940539aeaba.png" alt="alt text">
  <img src="https://user-images.githubusercontent.com/66253761/137334063-0cafc7ea-dac6-4be5-b055-19b13144538e.png" alt="alt text">
</p>

# Summary

Looking at the pictures, we can see that the images generated by the Pix2Pix model are more colorful and vibrant, while the CycleGAN’s images are lacking colors and a little washed out, but does color some objects a little, like sky and grass, up to an extent. 

Firstly, as the CycleGAN paper states, a conditional GAN (like Pix2Pix) would work better with paired images, while CycleGAN is usually used with unpaired images.

Secondly, training the CycleGAN model took around 4 times more than the Pix2Pix model, so the number of epoches it was trained for was lower (due to resource limitations), so that also may have contributed to the lack of learning of the model.

Overall we can see that while GANs are a great way to colorize B/W images, doing so with a CycleGAN model isn't the best idea as there are better models for this task.
