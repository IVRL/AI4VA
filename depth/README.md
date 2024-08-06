### Depth Estimation Challenge
![depth](https://github.com/IVRL/AI4VA/assets/16324609/0e431cd4-8fc2-40be-ab50-90d179db3ec3)

#### Overview

Our workshop features three challenges focusing on applying computer vision to visual arts, centered around the AI4VA dataset. This dataset, a first-of-its-kind benchmark, is designed to evaluate vision models’ ability to interpret comic art, focusing on segmentation, depth, and saliency. It includes detailed annotations of diverse comic styles, such as the toon-styled 'Placid & Muzo'.

#### Challenge Description

In the depth estimation challenge, participants are tasked with developing models to accurately estimate the depth of various elements within comic art images. The goal is to create models that can infer the relative distances of objects from the viewer, providing a depth map that accurately represents the scene's 3D structure.

#### Dataset

The AI4VA dataset consists of images with detailed depth annotations specific to the depth estimation task. These annotations include:

- **Depth Annotations**: Ground truth depth values for each segment in the images.
- **Variety of Comic Styles**: Depth annotations for two comic styles, challenging the model's ability to generalize across different visual representations.

The dataset covers a diverse array of comic art styles, providing a comprehensive challenge for model accuracy and generalization.

#### Step-by-Step Plan for Depth Ordering
1. **Setup Environment**

    Install necessary libraries (TensorFlow, PyTorch, OpenCV, etc.).
    ```bash
    pip install tensorflow torch torchvision opencv-python matplotlib
    ```
    Ensure GPU support for faster training if available.

2. **Download and Prepare Dataset**

    Download the images and the annotations from: [Google Drive](https://drive.google.com/drive/folders/1C5ER7Trz7I-oyzV7YndNZZ6UJMuNTH10?usp=sharing).

3. **Organize Dataset**

    Place the training and validation images into their respective folders. Verify the dataset structure and understand the contents.

4. **Data Exploration**

    Use the provided notebook `show_annotations_depth (1).ipynb` to explore the dataset and the annotations.
    Visualize a few images and their corresponding depth maps to understand the data.

5. **Baseline Model**

    Run the baseline model provided in `models/baseline_model.py`.
    Understand the preprocessing steps, model architecture, and evaluation metrics used.

6. **Data Preprocessing**

    Modify `scripts/data_preprocessing.py` to preprocess the data for depth ordering.

7. **Model Development**

    Enhance `scripts/train_model.py` to include a model architecture suitable for depth ordering.

8. **Model Training**

    Train the model using the enhanced script.
    Monitor training and validation accuracy to avoid overfitting.

9. **Model Evaluation**

    Use `scripts/evaluate_model.py` to assess the performance of your model on the validation set.

10. **Submission Preparation**

    Save your model's predictions and evaluation metrics in the `results/` directory.
    Follow the submission guidelines provided on the [Codalab page. ](https://codalab.lisn.upsaclay.fr/competitions/19857)
    Due to a server issue on Codalab, submissions are currently not being processed on Chrome and Edge browsers. Please use Firefox to submit your predictions (tested on Firefox 129.0) .

By following these steps, you should be able to complete the Depth Ordering Challenge effectively. If you need further assistance with any specific part, feel free to ask by creating an issue!

We encourage participants to explore different approaches, share their findings, and collaborate to push the boundaries of computer vision in the domain of visual arts. Happy coding, and may the best model win!
