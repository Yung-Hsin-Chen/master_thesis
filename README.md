# TrOCR meets CharBERT

## Installation
Dockerfiles are included. Thus, please ensure that Docker is installed.

1. Compile the report by navigating to the ```master_thesis``` folder, and then build the Docker.
    ```
    docker build -t master_thesis/document:v1.0 . -f Document/Dockerfile
    ```
2. Run the Docker image.
    ```
    docker run -it -v $PWD/Document:/work/Document master_thesis/document:v1.0
    ```
3. After getting into the Docker container, navigate to the ```Document``` folder.
    ```
    cd Document
    ```
4. Compile the ```.tex``` file to get the report pdf.
    ```
    pdflatex master_thesis_YH.tex
    ```