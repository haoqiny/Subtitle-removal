# FROM continuumio/miniconda3

# WORKDIR /root

# RUN apt-get update && apt-get install ffmpeg libgl1 -y

# COPY environment.yml .
# RUN conda env create -f environment.yml

# COPY E2FGVI/ E2FGVI/
# COPY main.py .
FROM continuumio/miniconda3

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y ffmpeg libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create Conda environment
COPY environment.yml .
RUN conda create -n pixa python=3.10 -y && conda clean -afy

# Activate environment and install conda dependencies
SHELL ["conda", "run", "-n", "pixa", "/bin/bash", "-c"]

# Install conda dependencies (excluding pip ones)
RUN conda install -n pixa -c defaults -c conda-forge \
    numpy ffmpeg pytorch torchvision \
    -y && conda clean -afy

# Install pip dependencies separately
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY E2FGVI/ E2FGVI/
COPY main.py .
RUN mkdir -p E2FGVI/release_model
COPY release_model/E2FGVI-HQ-CVPR22.pth E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth

# Entry point
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pixa", "python", "main.py"]
