from leanvector import IBMVectorDataEngine, VectorIndexes
import leanvector.test_credentials  # noqa
from cosaccess import CosAccessManager

print("Initializing IBMVectorDataEngine...")
ibm_vector_engine = IBMVectorDataEngine(vector_project='build test',
                                        apikey=test_credentials.apikey,
                                        bucket=test_credentials.bucket)
