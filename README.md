# `leanvector` Library for serverless vector indexing and search

The purpose of the `leanvector` library is to spin up stateless Python runtimes to store vector embeddings, create and manage vector indexes
and serve vector searches in a very lightweight fashion.
<br>

![](leanvector.png?raw=true)

It relies on Facebooks [FAISS](https://faiss.ai) library for the core vector indexing and search functions.

## Setup
```
pip install leanvector
```

## Example usage
```
from leanvector import VectorDataEngine, VectorIndexes

vector_engine = VectorDataEngine(vector_project='build test',
                                 s3endpoint=<your S3 endpoint>,
                                 s3accesskey=<your S3 Access Key>,
                                 s3secretkey=<your S3 Secret Key>,
                                 bucket=<your S3 Access Bucket>
                                 vectors=<your vector data as numpy array>)

vector_engine.create_index(VectorIndexes.HNSW)

D, I = vector_engine.search(<your query vector as numpy array>, VectorIndexes.HNSW)
```

## Demo
You can find a fully reproducible end-to-end demo in the [leanvector.ipynb](leanvector.ipynb) notebook here in this repository.

You can run these notebooks yourself using Jupyter as follows:

```bash
# Clone the Repository
git clone https://github.com/torsstei/leanvector.git

# Change directory
cd leanvector

# Set up your virtual environment
source ./setup_env.sh

# Install Jupyter
pip install jupyter

# Run Jupyter
jupyter notebook

```

## CosAccessManager method list
### Initialization
 * `VectorDataEngine(vector_project, bucket, s3endpoint, s3accesskey, s3secretkey, vectors = None, fvecs_file = None)` Constructor.
   * `vector_project`: You can store multiple vector data sets on the same S3 bucket in different sub folders identified by this vector project name
   * `bucket`: Your S3 bucket name where the vector data and indexes are to be stored
   * `s3endpoint` The endpoint of an S3 API compatible object storage
   * `s3accesskey` HMAC Access Key
   * `s3secretkey` HMAC Secret Key
   * `vectors` When provided this is ingested as the new vector data in this vector project
   * `fvecs_file` When provided this file is parsed and its vector data is ingested in this vector project. You must decide to provide vector data either as numpy array in `vectors` or as .vecs file on local disk in `fvecs_file`. When neither is provided the currently stored vector data set in the vector project is used.   
### Index Creation
 * `create_index(index_type)` Computes a vector index of the type specified in `index_type` (one of `leanvector.VectorIndexes`) and stores it in the S3 bucket.
### Vector Search
 * `search(query_vector, index_type = VectorIndexes.FLAT, k = 100)` Runs a vector search with the specified index (FLAT index by default). In parameter `k` you specify the number of nearest neighbors to return. At first search invocation of a certain index it is loaded and cached in memory from S3. So the first call can take longer. 
### Helper Methods
 * `cache_all_indexes()` Reads all indexes from S3 bucket and caches them in memory. This avoids longer latency at first search invocation of an index.
 * `get_index_size(index_type)` Returns the size of the index
 * `get_vector_array()` Returns the current vector data set as a numpy array

## Building and testing the library locally
### Set up Python environment
Run `source ./setup_env.sh` which creates and activates a clean virtual Python environment.
### Install the local code in your Python environment
Run `./_install.sh`.
### Test the library locally
1. Create a file `leanvector/test_credentials.py` with the S3 Endpoint, Bucket, Secret Key and Access Key:
```
endpoint=<your S3 endpoint>,
accesskey=<your S3 Access Key>,
secretkey=<your S3 Secret Key>,
bucket=<your S3 Access Bucket>
```
you can use the template file `leanvector/test_credentials.py.template`

2. Run `python leanvector/test.py`.

### Packaging and publishing distribution
1. Make sure to increase `version=...` in `setup.py` before creating a new package.
2. Run `package.sh`. It will prompt for user and password that must be authorized for package `cosaccess` on pypi.org.
