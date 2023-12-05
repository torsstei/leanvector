# ------------------------------------------------------------------------------
# Torsten Steinbach 2023
# torsten@steinbachnet.de
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import boto3
from enum import Enum
import faiss
import numpy as np
import os


class VectorIndexes(Enum):
    FLAT = "flat"
    LSH = "lsh"
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    PQ = "pq"
    IVFPQ = "ivfpq"


class VectorDataEngine:

    def __init__(self,
                 vector_project: str,
                 bucket: str,
                 s3endpoint: str,
                 s3accesskey: str,
                 s3secretkey: str,
                 fvecs_file: str = None):
        self._vector_project = vector_project
        self._prefix = "vector_data/" + self._vector_project.replace(" ", "_") + "/"
        self._vector_data_loaded = False
        self._s3endpoint = s3endpoint
        self._s3accesskey = s3accesskey
        self._s3secretkey = s3secretkey
        self._bucket = bucket
        self._vector_data: np.array = None
        if s3accesskey:
            self._s3client = boto3.client(service_name='s3', aws_access_key_id=self._s3accesskey,
                                          aws_secret_access_key=self._s3secretkey, endpoint_url=self._s3endpoint)
            self._s3resource = boto3.resource(service_name='s3', aws_access_key_id=self._s3accesskey,
                                              aws_secret_access_key=self._s3secretkey, endpoint_url=self._s3endpoint)

        os.makedirs("./.VectorDataEngine/" + self._prefix, exist_ok=True)
        self._loaded_indexes = {
            VectorIndexes.FLAT: None,
            VectorIndexes.LSH: None,
            VectorIndexes.HNSW: None,
            VectorIndexes.IVFFLAT: None,
            VectorIndexes.PQ: None,
            VectorIndexes.IVFPQ: None
        }

        if fvecs_file:
            self._s3resource.Object(self._bucket, self._prefix + "data.fvecs").put(Body=open(fvecs_file, 'rb'))

    def _load_vector_data(self):
        if not self._vector_data_loaded:
            tmpfile = "./.VectorDataEngine/" + self._prefix + "data.fvecs"
            self._s3client.download_file(self._bucket, self._prefix + "data.fvecs", tmpfile)
            tmp = np.fromfile(tmpfile, dtype='int32')
            self._vector_data = tmp.reshape(-1, tmp[0] + 1)[:, 1:].copy().view('float32')
            self._vector_data_loaded = True

    def get_vector_array(self):
        self._load_vector_data()
        return self._vector_data

    def create_index(self, index_type: VectorIndexes):
        self._load_vector_data()
        D = self._vector_data.shape[1]
        index = None
        if index_type == VectorIndexes.FLAT:
            index = faiss.IndexFlatIP(D)
        elif index_type == VectorIndexes.LSH:
            nbits = D * 8
            index = faiss.IndexLSH(D, nbits)
        elif index_type == VectorIndexes.HNSW:
            M = 64  # number of connections each vertex will have
            ef_search = 32  # depth of layers explored during search
            ef_construction = 64  # depth of layers explored during index construction
            # initialize index
            index = faiss.IndexHNSWFlat(D, M)
            # set efConstruction and efSearch parameters
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
        elif index_type == VectorIndexes.IVFFLAT:
            nlist = 128  # number of cells/clusters to partition data into
            quantizer = faiss.IndexFlatIP(D)  # how the vectors will be stored/compared
            index = faiss.IndexIVFFlat(quantizer, D, nlist)
            index.train(self._vector_data)  # we must train the index to cluster into cells
            index.nprobe = 4  # set how many of nearest cells to search
        elif index_type == VectorIndexes.PQ:
            m = 8  # number of sub vectors to split
            assert D % m == 0  # dimensions must dividable by the number of sub vectors
            nbits = 8  # number of bits per subquantizer, k* = 2**nbits
            index = faiss.IndexPQ(D, m, nbits)
            index.train(self._vector_data)
        elif index_type == VectorIndexes.IVFPQ:
            nlist = 128  # number of cells/clusters to partition data into
            nbits = 8  # when using IVF+PQ, higher nbits values are not supported
            m = 8  # number of sub vectors to split
            assert D % m == 0  # dimensions must dividable by the number of sub vectors
            quantizer = faiss.IndexFlatIP(D)  # how the vectors will be stored/compared
            index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits)
            index.train(self._vector_data)  # we must train the index to cluster into cells
            index.nprobe = 4  # set how many of nearest cells to search
        else:
            raise ValueError("Index type {} not supported".format(index_type))
        index.add(self._vector_data)  # This is the actual index creation operation
        self._loaded_indexes[index_type] = index  # We cache the index in memory once we have it
        # Now we perist and upload the index to S3
        idx_file = "./.VectorDataEngine/" + self._prefix + str(index_type) + ".index"
        faiss.write_index(index, idx_file)
        self._s3resource.Object(self._bucket, self._prefix + str(index_type) + ".index").put(Body=open(idx_file, 'rb'))

    def _load_index_data(self, index_type: VectorIndexes):
        if not self._loaded_indexes[index_type]:
            idx_file = "./.VectorDataEngine/" + self._prefix + str(index_type) + ".index"
            self._s3client.download_file(self._bucket, self._prefix + str(index_type) + ".index", idx_file)
            self._loaded_indexes[index_type] = faiss.read_index(idx_file)

    def search(self, query_vector: np.array, index_type: VectorIndexes = VectorIndexes.FLAT,
               k=100):  # k is the number of nearest neighbors to return
        self._load_index_data(index_type)
        return self._loaded_indexes[index_type].search(query_vector, k)

    def cache_all_indexes(self):
        for index in VectorIndexes:
            self._load_index_data(index)

    def get_index_size(self, index_type: VectorIndexes):
        self._load_index_data(index_type)
        return os.path.getsize("./.VectorDataEngine/" + self._prefix + str(index_type) + ".index")
