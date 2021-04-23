from azureml.core import Workspace, Dataset, Datastore
from azureml.core.environment import Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--create-workspace', dest='create_workspace',
                        action='store_const',
                        const=True, default=False,
                        help='Create a new workspace')

    parser.add_argument('--subscription-id', type=str,
                            dest='subscription-id',
                            default='',
                            help='Subscription id for creating a workspace')

    parser.add_argument('--create-compute', dest='create_compute', action='store_const',
                        const=True, default=False,
                        help='Create a new compoute instance')

    parser.add_argument('--gpus', dest='gpus',
                        type=int, default=2,
                        help='Number of gpus to provision')

    parser.add_argument('--create-env', dest='create_env', action='store_const',
                        const=True, default=False,
                        help='Create a new enviornment definition')

    parser.add_argument('--upload-data', dest='upload_data', action='store_const',
                        const=True, default=False,
                        help='Upload the data to Azure')

    parser.add_argument('--create-dataset', dest='create_dataset', action='store_const',
                        const=True, default=False,
                        help='Upload the data to Azure')

    parser.add_argument('--dataset-name', type=str,
                            dest='dataset_name',
                            default=False,
                            help='Dataset name, defaults to default dataset')

    parser.add_argument('--datastore-name', type=str,
                            dest='datastore_name',
                            default=False,
                            help='Datastore name, defaults to default datastore')

    parser.add_argument('--data-path', type=str,
                            dest='data_path',
                            default='/data',
                            help='Path in datastore')

    return parser.parse_args()

def create_ws(subscription_id):
    '''Creates an azure workspace'''

    subscription_id = subscription_id
    resource_group = 'birdsong_classification'
    workspace_name = 'birdsong_classification'

    workspace = Workspace.create(name=workspace_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group,
                   create_resource_group=True,
                   location='northcentralus'
                   )

    workspace.write_config(path='.azureml')

def create_compute(ws, gpus):
    '''Creates an azure compute cluster'''

    if gpus == 1:
        # # the name for the cluster
        compute_name = "gpu-cluster-NC6"
        # # the reference to the azure machine type
        vm_size = 'Standard_NC6_Promo'
    elif gpus == 2:
        # the name for the cluster
        compute_name = "gpu-cluster-NC12"
        # the reference to the azure machine type
        vm_size = 'Standard_NC12_Promo'
    elif gpus == 4:
        # the name for the cluster
        compute_name = "gpu-cluster-NC24"
        # the reference to the azure machine type
        vm_size = 'Standard_NC24_Promo'
    else:
        print(gpus, 'is not a valid number of GPUs.  No compute was created')
        return

    # define the cluster and the max and min number of nodes
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = 0,
                                                                max_nodes = 10)
    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

def create_env(ws):
    '''Creates an azureml enviornment'''

    # Create enviornment object
    env = Environment(name='birdsong-env-gpu')

    # define packages for image
    cd = CondaDependencies.create(pip_packages=['azureml-dataset-runtime[pandas,fuse]',
                                                'azureml-defaults',
                                                'tensorflow==2.4.0',
                                                'Pillow',
                                                'sklearn',
                                                'kapre',
                                                'sndfile',
                                                'librosa',
                                                'psutil'],
                                 conda_packages=['SciPy'])

    env.python.conda_dependencies = cd

    #Docker file
    dockerfile = r'''
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/o16n-base/python-assets:20210210.31228572 AS inferencing-assets

# Tag: cuda:11.0.3-devel-ubuntu18.04
# Env: CUDA_VERSION=11.0.3
# Env: NCCL_VERSION=2.8.3
# Env: CUDNN_VERSION=8.0.5.39

FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

USER root:root

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    iproute2 && \
    # Others
    apt-get install -y \
    build-essential \
    bzip2 \
    libbz2-1.0 \
    systemd \
    git \
    wget \
    cpio \
    pciutils \
    libnuma-dev \
    ibutils \
    ibverbs-utils \
    rdmacm-utils \
    infiniband-diags \
    perftest \
    librdmacm-dev \
    libibverbs-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libssl1.0.0 \
    linux-image-aws \
    linux-image-azure \
    linux-image-generic \
    linux-image-kvm \
    linux-image-lowlatency \
    linux-image-virtual \
    linux-image-gke \
    linux-image-oem \
    slapd \
    perl \
    ca-certificates \
    apt \
    p11-kit \
    libp11-kit0 \
    tar \
    libsndfile-dev \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Inference
# Copy logging utilities, nginx and rsyslog configuration files, IOT server binary, etc.
COPY --from=inferencing-assets /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=300
EXPOSE 5001 8883 8888

# Conda Environment
ENV MINICONDA_VERSION py37_4.9.2
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Open-MPI-UCX installation
RUN mkdir /tmp/ucx && \
    cd /tmp/ucx && \
	wget -q https://github.com/openucx/ucx/releases/download/v1.6.1-rc2/ucx-1.6.1.tar.gz && \
	tar zxf ucx-1.6.1.tar.gz && \
	cd ucx-1.6.1 && \
	./configure --prefix=/usr/local --enable-optimizations --disable-assertions --disable-params-check --enable-mt && \
	make -j $(nproc --all) && \
	make install && \
	rm -rf /tmp/ucx

# Open-MPI installation
ENV OPENMPI_VERSION 4.1.0
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --with-ucx=/usr/local/ --enable-mca-no-build=btl-uct --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Msodbcsql17 installation
RUN apt-get update && \
    apt-get install -y curl && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17

#Cmake Installation
RUN apt-get update && \
    apt-get install -y cmake

'''
    env.docker.base_image = None
    env.docker.base_dockerfile = dockerfile

    # Register environment to re-use later
    env = env.register(workspace = ws)

def upload_data(ws):
    '''upload data to the default azure datastore'''

    datastore = ws.get_default_datastore()

    # upload the mp3 files
    datastore.upload(src_dir='../data/audio_10sec',
                     target_path='/data/audio_10sec',
                     overwrite=False,
                     show_progress=True)

    # upload the npy files
    datastore.upload(src_dir='../data/npy',
                     target_path='/data/npy',
                     overwrite=False,
                     show_progress=True)

def create_dataset(ws, name, datastore, data_path):
    '''create the dataset object'''

    # get the datastore
    if datastore:
        datastore = Datastore.get(ws, datastore)
    else:
        datastore = ws.get_default_datastore()

    # define dataset
    dataset = Dataset.File.from_files(path=(datastore, data_path))

    # register the dataset for future use
    dataset = dataset.register(workspace=ws,
                               name=name,
                               create_new_version=True)

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    print('ARGS:', args)

    # Logic
    if args.create_workspace:
        print('Creating workspace...')
        create_ws(args.subscription_id)

    # get the workspace
    ws = Workspace.from_config()

    if args.create_compute:
        print('Creating Compute...')
        create_compute(ws, args.gpus)

    if args.create_env:
        print('Creating Enviornment...')
        create_env(ws)

    if args.upload_data:
        print('Uploading Data...')
        upload_data(ws)

    if args.create_dataset:
        print('Defining Dataset...')
        create_dataset(ws, args.dataset_name, args.datastore_name, args.data_path)

    print('Done.')
