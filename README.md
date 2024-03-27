# TrOCR meets CharBERT - On-Going

## Table of Contents
- [TrOCR meets CharBERT - On-Going](#trocr-meets-charbert---on-going)
  - [Table of Contents](#table-of-contents)
  - [Data](#data)
  - [Get Report](#get-report)
  - [Run Source Codes](#run-source-codes)

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
2. Run the container in an interactive mode.
   
   1. Run ```processor.py``` to prepare the data loaders.
        ```
        python ./src/processor/precessor.py
        ```
    2. Start training or evaluation by modifying the ```main.py```.
        ```
        python ./src/main.py
        ```