# style-content-cnn
Style and Content CNNs used in GANILLA [link](https://github.com/giddyyupp/ganilla).

## Download Models and Data
- You can download style and content models using following [link](https://drive.google.com/open?id=1Lz4uLVdtqtBqMj2RzljKhrLNEQ87NYPg)
- We also shared test images used for content CNN tests. 

## How to use these networks

### Requirements
Python2 (if you fix print statements then im pretty sure it will work with Python3 also.)

Pytorch versions > 0.4.0 should work. Install the requirements with `pip install -r requirements.txt`. 

### Style CNN
- Style CNN is trained using original illustrations(source, style set) images. Also as a negative class, natural photographs added.
So in our case there are 11 classes, 10 from illustrations (AS, KP etc.) and natural images from SUN Dataset [link](https://groups.csail.mit.edu/vision/SUN/).
- In order to test Style CNN, you should use translated images. Here we used CycleGan test images (751 image) as source to style transfer.
So we feed the same images to all GANILLA models, and get outputs. These images are then used as test set.

#### Train&Test
You need to create below folder structure to train and test style cnn. I put sample counts in our case in the parentheses.

    .
    ├── dataset_style
        ├── train           # Contains any number of illustrators, in our case we got 10 illustrators. You can check CLASS_LIST var in style/config.py 
        │   ├── axel (500)         
        │   ├── korky (300)       
        │   ├── polacco (650)
        |   └── ...
        ├── test            # same folders with train except you don't need to put photo class here, since we are not interested about that one.
            ├── axel   (751 for all)       
            ├── korky         
            ├── polacco 
            └── ...   

After you prepare above folder structure, and update necessary vars in `style/config.py` you can train the style cnn using `python style/train.py`.

After training finishes, test with `python style/test.py`. It will print class based f1 scores and average accuracy to the terminal.

### Content CNN
- Content CNN is trained using selected categories in SUN Dataset. Also as a negative class, illustrations added.
So in our case there are 11 classes, 10 from SUN Dataset (beach, valley etc.) and one illustrations class 
(samples taken from illustrators which are not used in experiments!).
- In order to test Content CNN, selected images in SUN Dataset is used as source domain to style transfer. 
We shared these images in above Google Drive link. So these images are used as source to style transfer, then outputs are used 
as test set.

#### Train&Test
You need to create below folder structure to train and test content cnn. I put sample counts in our case in the parentheses. 

    .
    ├── dataset_content
        ├── train              # Contains natural image categories, in our case we got 10 categories from SUN dataset. You can check CLASS_LIST var in content/config.py 
        │   ├── all_ils (2963) # please put unused illustarations (eg. if you trained style transfer models for axel and korky, you shouldn't add them here.)
        │   ├── beach   (1144)    
        │   ├── forest  (377)
        |   └── ...
        ├── test            # here test folder structure is tricky. We want to evaluate each style separately.
            ├── axel          
            │  ├── beach  (50)       
            │  ├── forest (50)
            │  └── ...   
            ├── korky 
            │  ├── beach   (50)         
            │  ├── forest  (50) 
            │  └── ...   
            
After you prepare above folder structure, and update necessary vars (such as `train_dir_path` and `train_path`) in `content/config.py` you can train the content cnn using `python content/train.py`.

After training finishes, test with `python content/test.py` (dont forget to update vars such as `test_dir_path` and `test_path`). It will print class based f1 scores and average accuracy to the terminal.


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
