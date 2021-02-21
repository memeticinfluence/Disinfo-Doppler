FROM jupyter/datascience-notebook
WORKDIR /home/jovyan/work
COPY --chown=${NB_UID}:${NB_GID} . /home/jovyan/work
RUN mkdir /home/jovyan/work/data
RUN pip install --requirement requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
