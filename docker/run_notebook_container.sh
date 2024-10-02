docker run --gpus all -it --rm \
        -v /mnt/pccfs2/backed_up/mckay/thesis:/thesis \
        -v /mnt/pccfs2/backed_up/models:/models \
        -p 8000:8000 \
        mckay-thesis:latest