# style-content-cnn
Style and Content CNNs used in GANILLA [link](https://github.com/giddyyupp/ganilla).

## Download Models and Data
- You can download style and content models using following [link](https://drive.google.com/open?id=1Lz4uLVdtqtBqMj2RzljKhrLNEQ87NYPg)
- We also shared test images used for content CNN tests. 

## How to use these networks
### Style CNN
- Style CNN is trained using original illustrations(source, style set) images. Also as a negative class, natural photographs added.
So in our case there are 11 classes, 10 from illustrations (AS, KP etc.) and natural images from SUN Dataset [link](https://groups.csail.mit.edu/vision/SUN/).
- In order to test Style CNN, you should use translated images. Here we used CycleGan test images (751 image) as source to style transfer.
So we feed the same images to all GANILLA models, and get outputs. These images are then used as test set.

- Content CNN is trained using selected categories in SUN Dataset. Also as a negative class, illustrations added.
So in our case there are 11 classes, 10 from SUN Dataset (beach, valley etc.) and one illustrations class 
(samples taken from illustrators which are not used in experiments!).
- In order to test Content CNN, selected images in SUN Dataset is used as source domain to style transfer. 
We shared these images in above Google Drive link. So these images are used as source to style transfer, then outputs are used 
as test set.




## Citation
If you use this code for your research, please cite our papers.
```
@article{hicsonmez2020ganilla,
  title={GANILLA: Generative adversarial networks for image to illustration translation},
  author={Hicsonmez, Samet and Samet, Nermin and Akbas, Emre and Duygulu, Pinar},
  journal={Image and Vision Computing},
  pages={103886},
  year={2020},
  publisher={Elsevier}
}

@inproceedings{Hicsonmez:2017:DDN:3078971.3078982,
 author = {Hicsonmez, Samet and Samet, Nermin and Sener, Fadime and Duygulu, Pinar},
 title = {DRAW: Deep Networks for Recognizing Styles of Artists Who Illustrate Children's Books},
 booktitle = {Proceedings of the 2017 ACM on International Conference on Multimedia Retrieval},
 year = {2017}
}
```