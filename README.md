## TrOCR meets CharBERT - On-Going
- [TrOCR meets CharBERT - On-Going](#trocr-meets-charbert---on-going)
- [Data](#data)
- [Get Report](#get-report)
- [Run Source Codes](#run-source-codes)
- [Folder Structure](#folder-structure)

## Data
Data used in the project can be downloaded online.
1. [IAM Handwritten Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
2. [GW Handwritten Dataset](https://fki.tic.heia-fr.ch/databases/washington-database)

## Get Report
Dockerfiles are included. Thus, please ensure that Docker is installed.

1. Compile the report by navigating to the ```master_thesis``` folder, and then build the Docker.
    ```
    docker build -t master_thesis/report:v1.0 . -f report/Dockerfile
    ```
2. Run the Docker image.
    ```
    docker run -it -v $PWD/report:/work/report master_thesis/report:v1.0
    ```
3. After getting into the Docker container, navigate to the ```report``` folder.
    ```
    cd report
    ```
4. Compile the ```.tex``` file to get the report pdf.
    ```
    pdflatex master_thesis_YH.tex
    ```
5. Compile the bibliography.
    ```
    bibtex master_thesis_YH
    ```
6. Compile the ```.tex``` file to get the report pdf.
    ```
    pdflatex master_thesis_YH.tex
    ```

## Run Source Codes
1. Navigate to ```master_thesis``` folder, put data in ```data/raw/data```, and build the docker image.
    ```
    docker build -t master_thesis/ocr_correction:v.1.0 . -f Dockerfile
    ```
2. Run the container in an interactive mode. Start training or evaluation by modifying the ```main.py```.
    ```
    python ./src/main.py
    ```

## Folder Structure
```
master_thesis
│--- README.md        <- Contains an overview of the project, setup 
│                        instructions, and any additional information 
│                        relevant to the project.
│--- Dockerfile 
│--- run_script.sh    <- A shell script for executing common tasks, 
│                        such as setting up the environment, starting 
│                        a training run, or evaluating models.  
│--- setup.py
│--- requirements.txt
│--- config           <- Directory containing configuration files for
│                        models, training processes, or application 
│                        settings.
│--- data             <- Datasets used in the thesis.
│--- models           <- Contains saved models.
│--- notebook         <- Jupyter notebooks for exploratory data analysis.
│--- report           <- Stores the final report.
│--- results          <- Contains output from model evaluations, 
│                        including metrics.
└─── src              <- Source code for the project.
    │--- processor    <- Code related to data preprocessing and  
    │                    preparing raw data for training or evaluation.
    │--- models       <- Definitions of the machine learning models used
    │                    in the thesis
    │--- train        <- Scripts and modules for training models.
    │--- eval         <- Scripts and modules for evaluating models.
    │--- utils        <- Utility functions and classes that support 
    │                    various tasks across the project, such as data 
    │                    loading, metric calculation, and visualization 
    │                    tools.
    └─── tests        <- Automated tests for the codebase
```