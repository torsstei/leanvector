import os
os.chdir("leanvector")

from leanvector import VectorDataEngine, VectorIndexes
import test_credentials  # noqa
import numpy as np

print("Initializing VectorDataEngine...")
vector_engine = VectorDataEngine(vector_project='build test',
                                 s3endpoint=test_credentials.endpoint,
                                 s3accesskey=test_credentials.accesskey,
                                 s3secretkey=test_credentials.secretkey,
                                 bucket=test_credentials.bucket)
