# Saliency Estimation Challenge
<img width="1024" alt="saliency" src="https://github.com/IVRL/AI4VA/assets/16324609/6284eecc-ef90-4298-85a6-a172c155b657">

#### Overview

Our workshop features three challenges focusing on applying computer vision to visual arts, centered around the AI4VA dataset. This dataset, a first-of-its-kind benchmark, is designed to evaluate vision models’ ability to interpret comic art, focusing on segmentation, depth, and saliency. It includes detailed annotations of diverse comic styles, such as the toon-styled 'Placid & Mizo'.

#### Challenge Description

In the saliency estimation challenge, participants are tasked with developing models to accurately predict human visual attention within comic art images. The goal is to create models that can generate saliency maps, highlighting regions in the images that are likely to attract human attention.

#### Dataset

The AI4VA dataset consists of images with detailed saliency annotations specific to the saliency estimation task. These annotations include:

- **Saliency Maps**: Ground truth maps indicating the regions of the image that draw the most human attention.
- **Variety of Comic Styles**: Saliency annotations for a range of comic styles, challenging the model's ability to generalize across different visual representations.

The dataset covers a diverse array of comic art styles, providing a comprehensive challenge for model accuracy and generalization.

#### Instructions

1. **Download the Dataset**: Follow the instructions in the `data/README.md` to download and set up the AI4VA dataset.

2. **Data Exploration**: Use the `notebooks/data_exploration.ipynb` to understand the dataset structure and visualize the saliency annotations.

3. **Baseline Model**: Start with the `notebooks/baseline_model.ipynb` notebook, which provides a simple baseline saliency estimation model. This notebook will guide you through the initial steps of data preprocessing, model training, and evaluation.

4. **Develop Your Model**:
    - **Preprocess the Data**: Use the `scripts/data_preprocessing.py` script to prepare the data for training.
    - **Train Your Model**: Modify and enhance the `scripts/train_model.py` to train your saliency estimation model. Experiment with different architectures and hyperparameters.
    - **Evaluate Your Model**: Use the `scripts/evaluate_model.py` to assess the performance of your model on the validation set. Focus on metrics such as Area Under Curve (AUC) and Normalized Scanpath Saliency (NSS).

5. **Submit Your Results**: Save your model's predictions and evaluation metrics in the `results/` directory. Follow the submission guidelines provided in the `README.md`.

#### Evaluation

Your submissions will be evaluated based on the following criteria:

- **Accuracy**: How accurately does your model predict human visual attention in the comic art images?
- **Generalization**: How well does your model perform across different comic styles in the dataset?
- **Innovation**: Novel approaches and innovative techniques will be highly regarded.

#### Resources
Baselines : https://github.com/matthias-k/DeepGaze (DeepGaze IIE)
https://github.com/samyak0210/saliency 
https://github.com/IVRL/Tempsal

- **Baseline Notebooks**: `notebooks/baseline_model.ipynb`
- **Scripts**: `scripts/data_preprocessing.py`, `scripts/train_model.py`, `scripts/evaluate_model.py`
- **Evaluation Metrics**: AUC, NSS

We encourage participants to explore different approaches, share their findings, and collaborate to push the boundaries of computer vision in the domain of visual arts. Happy coding, and may the best model win!
