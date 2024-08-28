
![header-ai4va](https://github.com/IVRL/AI4VA/assets/16324609/8fffabcf-72a0-4d08-9328-69798105a027)


# AI for Visual Arts Challenges (AI4VA) on Depth and Saliency

https://sites.google.com/view/ai4vaeccv2024/participation

Welcome to the AI for Visual Arts Challenges repository! This repository contains the resources you need to participate in the challenge, including example notebooks, baseline models, and instructions.

**Important Notice: Submission Issue on Codalab Challenge**

**_Due to a server issue on Codalab, submissions are currently not being processed on Chrome and Edge browsers. Please use Firefox to submit your predictions (tested on Firefox 129.0) ._**


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

## NEW !! Few FAQs about the challenges: 
1) The final score of the competition is determined by the rankings in both the Development stage and the Code submission stage, which should be identical. In other words, the rankings should not differ between these two stages.
2) The top two teams from the Development stage will be required to submit their code through a public GitHub repository. The link to this repository should be sent to us via email, along with a technical report detailing your model. Specifics on the report can be found at: https://sites.google.com/view/ai4vaeccv2024/participation, under Challenge Participation. We will review and run the submitted code to verify that the rankings in the Development and Code Review stages match. To ensure fairness and replicate your original environment, it is crucial that the technical report includes information on all packages, libraries with version details, the platform used, the number of GPUs employed, and their specifications.
3) The leaderboard visible on CodaLab reflects the rankings. Ideally, the leaderboard rankings in the Development stage should match those in the Code Review stage. In the event of a tie, both top-performing teams will be awarded.
4) Certificates will be awarded to the top two teams, with additional prizes given only to the top-performing team.

## License

The provided model and datasets are available for unrestricted use in personal research, non-commercial, and not-for-profit endeavours. For any other usage scenarios, kindly contact the AI4VA organisers via Email: ai4vaeccv2024-organizers@googlegroups.com, providing a detailed description of your requirements. 
