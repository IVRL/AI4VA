
![header-ai4va](https://github.com/IVRL/AI4VA/assets/16324609/8fffabcf-72a0-4d08-9328-69798105a027)


# AI for Visual Arts Challenges (AI4VA)

Welcome to the AI for Visual Arts Challenges repository! This repository contains the resources you need to participate in the challenge, including example notebooks, baseline models, and instructions.

**Important Notice: Submission Issue on Codalab Challenge**

**_Due to a server issue on Codalab, submissions are currently not being processed. We are working to resolve this as quickly as possible. Thank you for your patience and understanding._**


## Repository Structure

We have two challenge tracks: **depth** and **saliency estimation**. Each track has the following structure:

- `data/`: Contains training and validation images and ground truth data.
- `notebooks/`: Contains Jupyter notebooks for data exploration.
- `scripts/`: Contains Python scripts for evaluation and metrics.
- `models/`: A folder to place your models.
- `results/`: Contains your models' predictions.
- `requirements.txt`: Lists Python dependencies.
- `README.md`: This file, provides a task-specific overview and instructions.

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/IVRL/AI4VA.git
    cd AI4VA
    ```

2. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download and organize the data:**
   We provide the data for each task in this Google Drive folder :  https://drive.google.com/drive/folders/1wkZrOFQx3LZnG_rEc_js1WvNf5HHcGtn?usp=sharing
   Follow the instructions on the task page for more details.
5. **Explore the data:**
    Open and run `show_annotations.ipynb`.

6. **Run the baseline model:**
    Open and run `notebooks/baseline_model.ipynb`.

Please read the instructions in the respective challenge folders for more details.

## License

The provided model and datasets are available for unrestricted use in personal, research, non-commercial, and not-for-profit endeavours. For any other usage scenarios, kindly contact the AI4VA organisers via Email: ai4vaeccv2024-organizers@googlegroups.com, providing a detailed description of your requirements. 
