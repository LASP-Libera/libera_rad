# Libera Radiometer L1b Algorithm

The libera_rad package processes L1A data and geolocation files into the L1B radiometer data product. The goal of this
algorithm is to convert radiometer DNs into radiance units. In this process, the radiometer data is calibrated for
gain/noise, downsampled, and associated with geolocation information. For more information on the algorithm, see
internal documentation at https://lasp.colorado.edu/galaxy/x/RJNbEg. A data flow diagram can be found at:
https://lasp.colorado.edu/galaxy/x/YQZgD

## Environment Setup

Using a virtual environment is STRONGLY recommended for ensuring proper dependency management. For instructions on how
to do this, see the libera utils developer's guide:
https://libera-utils.readthedocs.io/en/latest/developer-docs/dev_environment_setup.html

To step through the algorithm process with supporting plots and data insights, run
learning_notebooks/l1b_algorithm.ipynb.

To create the production algorithm docker image and the test docker image, run:

```bash
docker-compose build
```

To run either image, run:

```bash
docker run libera-rad-test
```

```bash
docker run libera-radiometer <cli arguments>
```
