# Saliency Estimation Challenge
<img width="1024" alt="saliency" src="https://github.com/IVRL/AI4VA/assets/16324609/6284eecc-ef90-4298-85a6-a172c155b657">

#### Overview

Our workshop features three challenges focusing on applying computer vision to visual arts, centered around the AI4VA dataset. This dataset, a first-of-its-kind benchmark, is designed to evaluate vision models’ ability to interpret comic art, focusing on segmentation, depth, and saliency. It includes detailed annotations of diverse comic styles, such as the toon-styled 'Placid & Muzo'.

#### Challenge Description

In the saliency estimation challenge, participants are tasked with developing models to accurately predict human visual attention within comic art images. The goal is to create models that can generate saliency maps, highlighting regions in the images that are likely to attract human attention. Saliency estimation in the comics domain presents a unique set of challenges due to their complex nature, variety of artistic styles and abstractions.

#### Dataset

The AI4VA dataset consists of images with saliency annotations specific to the saliency estimation task. These annotations include:

- **Saliency Maps**: Ground truth maps indicating the regions of the image that draw the most human attention. In these grayscale maps, brighter regions indicate a higher attention than the darker regions.
- **Variety of Comic Styles**: The two different comic styles, challenge the model's ability to generalize across different visual representations. "Yves le Loup" has a realistic style while "Placid et Muzo" has a simplistic style.

The dataset covers different comic art styles, providing a comprehensive challenge for model accuracy and generalization.

#### Instructions

1. **Download the Dataset**: Download the images and ground truth maps from : https://drive.google.com/file/d/1cNivYQCyEz6AncLTJnAfrgkGRyMsVmm_/view?usp=sharing
   
   Place them under the data/ folder into the train and validation folders respectively.

3. **Data Exploration**: Use the `show_annotations_saliency.ipynb` to understand the dataset structure and visualize the saliency annotations.
Make sure that `"comic_images_folder` and `annotations_folder` are correct.
4. **Baseline Model** (Optional): Start with the `notebooks/baseline_model.ipynb` notebook, which provides a baseline saliency estimation model (DeepGazeIIE). This notebook provides an inference for the saliency estimation task.

5. **Develop Your Model**:
    - **Preprocess the Data**: Inspect the `model/custom_dataset.py` which prepares the data for training and validation.
    - **Train Your Model**: Modify and enhance the `scripts/train.py` to train your saliency estimation model. Experiment with different architectures and hyperparameters. You can implement your model in `models/model.py` or import any baseline.
    - **Evaluate Your Model**: Use the `scripts/evaluate_model.py` to assess the performance of your model on the validation set. Aim to improve all metrics such as Area Under Curve (AUC), Correlation Coefficient (CC) and Kullbeck-Leibler Divergence (KLD). The implementations of these metrics are provided in `scripts/metrics.py`.

6. **Submit Your Results**: Save your model's predictions on the test dataset in the `results/` directory. Follow the submission guidelines provided on the [Codalab page](https://codalab.lisn.upsaclay.fr/competitions/19855) .

#### Evaluation

Your submissions will be evaluated based on the following criteria:

- **Accuracy**: How accurately does your model predict human visual attention in the comic art images?
- **Generalization**: How well does your model perform across different comic styles in the dataset?
- **Innovation**: Novel approaches and innovative techniques will be highly regarded.

#### Resources
Baselines : 
https://github.com/matthias-k/DeepGaze (DeepGaze IIE)

https://github.com/samyak0210/saliency 

https://github.com/IVRL/Tempsal

- **Baseline Notebooks**: `notebooks/baseline_model.ipynb`
- **Scripts**:  `scripts/train.py`, `scripts/evaluate_model.py`
- **Evaluation Metrics**: AUC, CC, KLD

We encourage participants to explore different approaches, share their findings, and collaborate to push the boundaries of computer vision in the domain of visual arts. Happy coding, and may the best model win!
