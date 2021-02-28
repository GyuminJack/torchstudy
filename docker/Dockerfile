ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

RUN conda install -y pytorch torchvision -c pytorch
RUN conda install -y torchtext gensim tqdm

