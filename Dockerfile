FROM runpod/pytorch:1.4.1-cu1300-torch2121-ubuntu2404

# Install system dependencies if needed
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        p7zip-full \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /root/codesage_ft

COPY data/primevul.7z data

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install -r --no-cache-dir /tmp/requirements.txt

# Keep default RunPod entrypoint (starts SSH and JupyterLab)
COPY startup.sh .
RUN chmod +x startup.sh
CMD ["./startup.sh"]