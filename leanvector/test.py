import os
os.chdir("leanvector")

from leanvector import VectorDataEngine, VectorIndexes
import test_credentials  # noqa
import numpy as np
import datetime

"""
from contextlib import closing
import urllib.request as request
import shutil
import tarfile

with closing(request.urlopen('ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz')) as r:
    with open('siftsmall.tar.gz', 'wb') as f:
        shutil.copyfileobj(r, f)
tar = tarfile.open('siftsmall.tar.gz', "r:gz")
tar.extractall()
"""

vector_data = np.random.uniform(low=0.0, high=256.0, size=(1000,128))

print("Initializing VectorDataEngine...")
vector_engine = VectorDataEngine(vector_project='build test',
                                 s3endpoint=test_credentials.endpoint,
                                 s3accesskey=test_credentials.accesskey,
                                 s3secretkey=test_credentials.secretkey,
                                 bucket=test_credentials.bucket,
                                 vectors=vector_data)

xb = vector_engine.get_vector_array()
print(xb[:2])

vector_engine.create_index(VectorIndexes.FLAT)
vector_engine.create_index(VectorIndexes.LSH)
vector_engine.create_index(VectorIndexes.HNSW)
vector_engine.create_index(VectorIndexes.IVFFLAT)
vector_engine.create_index(VectorIndexes.PQ)
vector_engine.create_index(VectorIndexes.IVFPQ)

query = np.random.uniform(low=0.0, high=256.0, size=(1,128))
before_time = datetime.datetime.now()
D, I = vector_engine.search(query, VectorIndexes.FLAT)
print("Flat Index Search time: {} milliseconds".format((datetime.datetime.now() - before_time).total_seconds()*1000))
baseline = I[0].tolist()

before_time = datetime.datetime.now()
D, I = vector_engine.search(query, VectorIndexes.LSH)
print("LSH Index Search time: {} milliseconds".format((datetime.datetime.now() - before_time).total_seconds()*1000))
print("LSH Index Recall Rate: {} %".format(np.array(baseline)[np.in1d(baseline, I).tolist()].size / np.array(baseline).size * 100))

before_time = datetime.datetime.now()
D, I = vector_engine.search(query, VectorIndexes.HNSW)
print("HNSW Index Search time: {} milliseconds".format((datetime.datetime.now() - before_time).total_seconds()*1000))
print("HNSW Index Recall Rate: {} %".format(np.array(baseline)[np.in1d(baseline, I).tolist()].size / np.array(baseline).size * 100))

before_time = datetime.datetime.now()
D, I = vector_engine.search(query, VectorIndexes.IVFFLAT)
print("IVFFLAT Index Search time: {} milliseconds".format((datetime.datetime.now() - before_time).total_seconds()*1000))
print("IVFFLAT Index Recall Rate: {} %".format(np.array(baseline)[np.in1d(baseline, I).tolist()].size / np.array(baseline).size * 100))

before_time = datetime.datetime.now()
D, I = vector_engine.search(query, VectorIndexes.PQ)
print("PQ Index Search time: {} milliseconds".format((datetime.datetime.now() - before_time).total_seconds()*1000))
print("PQ Index Recall Rate: {} %".format(np.array(baseline)[np.in1d(baseline, I).tolist()].size / np.array(baseline).size * 100))

before_time = datetime.datetime.now()
D, I = vector_engine.search(query, VectorIndexes.IVFPQ)
print("IVFPQ Index Search time: {} milliseconds".format((datetime.datetime.now() - before_time).total_seconds()*1000))
print("IVFPQ Index Recall Rate: {} %".format(np.array(baseline)[np.in1d(baseline, I).tolist()].size / np.array(baseline).size * 100))
