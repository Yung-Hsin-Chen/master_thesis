# TrOCR meets CharBERT

## Installation
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