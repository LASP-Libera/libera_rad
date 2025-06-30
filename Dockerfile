## Dockerfile that installs libera_rad and its dependencies

ARG BUILDPLATFORM=linux/amd64

# libera-rad
# ----------
FROM --platform=${BUILDPLATFORM} python:3.11-slim AS libera-rad
USER root

# Location for Core package installation location. This can be used later by images that inherit from this one
ENV LIBERA_RAD_DIRECTORY=/opt/libera
WORKDIR $LIBERA_RAD_DIRECTORY

# Turn off interactive shell to suppress configuration errors
ARG DEBIAN_FRONTEND=noninteractive

# Install
# libpq so we can install psycopg2
# curl so we can install poetry
# gcc because it's often required for python package installations
RUN apt-get update && apt-get install -y libpq-dev curl gcc pkg-config libhdf5-dev

# Create virtual environment and permanently activate it for this image
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# This adds not only the venv python executable but also all installed entrypoints to the PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Upgrade pip to the latest version because poetry uses pip in the background to install packages
RUN pip install --upgrade pip

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
# Add poetry to path
ENV PATH="$PATH:/root/.local/bin"

# Copy necessary files over (except for dockerignore-d files)
COPY libera_rad $LIBERA_RAD_DIRECTORY/libera_rad
COPY pyproject.toml $LIBERA_RAD_DIRECTORY
COPY README.md $LIBERA_RAD_DIRECTORY
COPY LICENSE.txt $LIBERA_RAD_DIRECTORY

# This is so stupid but it fixes known a bug in docker build
# https://github.com/moby/moby/issues/37965
RUN true

# Install libera_rad and all its (non-dev) dependencies according to pyproject.toml
RUN poetry install --only main
# Define the entrypoint of the container. Passing arguments when running the
# container will be passed as arguments to the function
ENTRYPOINT ["libera-rad"]


# libera-rad-test
# ---------------
FROM libera-rad AS libera-rad-test

# Install dev dependencies (not installed in libera-rad image)
RUN poetry install

# Copy tests over
COPY tests $LIBERA_RAD_DIRECTORY/tests

# Set entrypoint
ENTRYPOINT ["pytest", "--cov=libera_rad", "--cov-report=xml:coverage.xml", "--junit-xml=junit.xml"]
