# eager_gradcam_tf
Grad-CAM implementation for TensorFlow 2.0 in Eager Execution mode.

To use you can put the script within your project folder and import it into a Jupyter Notebook or your IDE of preference. In the near future I will make it possible to build a pip wheel.

## Quick start
Use `grad-cam` function to view gradients and gradient/photo overlay.

```
grad_cam(image, model, image_dims, return_switch)
          |      |          |            |
          |      |          |            'gradients' to return list of gradients, 'maps' to return list of feature maps
          |      |          |
          |      |          tuple containing image dimensions e.g. (299, 299)
          |      |
          |      Model or Sequential object with eager execution enabled
          |
          path to image
```

## Notebooks
Feel free to play around with the notebooks I provided, they describe most of the steps I went through to figure out the best possible way to allow Grad-CAM to work in eager mode.

## Disclaimer

This implementation has been tested on Inception V3 architecture in TensorFlow 2.0 with Keras API.
With that being said I cannot guarantee that this implementation is absolutely accurate and will be happy to hear from the community to get suggestions of improvements to this project.

You can read the original paper by Ramprasaath R. Selvaraju, Michael Cogswell, et al. at https://arxiv.org/abs/1610.02391
