# CombPlex

![CombPlex's logo](https://github.com/liorbensha/Combplex/assets/89635508/8f2ff95d-770d-4faa-8724-abfb464369da)

**CombPlex** (COMBinatorial multiPLEXing) is a combinatorial staining platform integrated with an algorithmic framework, enabling a significant expansion in the number of proteins that can be visualized *in-situ*.

## CombPlex Demonstration Notebook
Explore the `CombPlex_demo.ipynb` Jupyter Notebook for a guided walkthrough that illustrates our method, **CombPlex**. We suggest downloading our Git directory, placing it within your Google Drive folder, and then opening `CombPlex_demo.ipynb` using Google Colab.

## System requirements
The code is compatible with Python 3.9 and PyTorch 2.2, and GPU access is required.
You can create an anaconda environment called **CombPlex** with the required dependencies by running:
```
conda env create -f requirements.yml
conda activate CombPlex
```

## Data Preparation
To get started, ensure that your training, validation, and test datasets are placed within the `datasets` folder.

Within each dataset folder, you should create subfolders for each FOV. Inside these FOV subfolders, you can organize the images related to that particular FOV, either single-protein images or compressed images. It's essential to maintain identical filenames across all FOV folders and ensure that all image files are in the TIF format.
```
<dataset_name>/
├── <FOV_name_1>/
│   ├── <image_name_1.tif>
│   ├── <image_name_2.tif>
│   │   ├── ...
├── <FOV_name_2>/
│   ├── <image_name_1.tif>
│   ├── <image_name_2.tif>
│   │   ├── ...
├── ...
```
**Note:** You might find the demo datasets helpful for understanding the expected data structure.

## Generating Compression Matrices CSV File
To create the required compression matrices, begin by utilizing the `compression_matrix_builder.py` file. Within this script, complete the specified parameters and execute it. The result will be an CSV file situated within the compression_matrices f`older. Notice you need to input filenames manually. Additionally, you have the option to modify the training or test matrix based on your specific requirements.

**Note:** For guidance and reference, you can explore the compression matrix files we've already generated for the experiments conducted in our paper

## Models Training Procedure
1. Follow the data preparation instructions outlined earlier.
2. Create your compression matrices CSV file as explained.
3. Customize the `config/config_train.yaml` according to your preferences.
4. Initiate models training by executing:
    ```
    train.py "Decompression masking network"
    train.py "Value reconstruction network"
    ```
5. If you wish to conduct an ensemble run, you can potentially iterate through steps 3 and 4 for multiple runs.

### Output files:
Upon completion of the training process, your trained models will be located within the `pretrained_models/{model_name}` directory. We suggest selecting the most recent optimally trained model. The term "optimal" denotes a model that achieved the lowest loss in the validation score thus far. However, you have the option to review performances and losses via W&B (Weights & Biases) and make an alternative decision if needed.

Once you've chosen your preferred trained model, kindly duplicate it into the `pretrained_models` folder and rename it to: `{model_name}_optimal.pt`.

## Model Inference Procedure
1. Follow the data preparation instructions outlined earlier.
3. Customize the `config/config_inference.yaml` according to your preferences.
4. Execute the `inference.py` file to run the model's inference on the designated test set.

### Output files:
Upon completing the inference process, the `results/results_folder_name` directory will contain the following files:

1. Compressed images: `{compressed_image_name}.tif`
2. Final reconstructed single-protein images: `recon_{protein_name}_{f1_score}.tif`
3. Ground truth single-protein images (if provided): `{protein_name}.tif`
4. A CSV file containing a table of F1 scores: `F1 scores results - {results_folder_name}.csv` and a corresponding box plot.
5. A CSV file containing a table of Pearson correlation coefficients between prediction and ground truth intensities: `Pearson correlations - {results_folder_name}.csv` and a corresponding box plot.

**Note:** Evaluation processes are feasible solely when actual ground truth single-protein images are available.
