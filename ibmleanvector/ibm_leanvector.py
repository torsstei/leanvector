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

import ibm_boto3
from ibm_botocore.client import Config
from cosaccess import CosAccessManager
from leanvector import VectorDataEngine

class IBMVectorDataEngine(VectorDataEngine):

    def __init__(self, vector_project:str, bucket:str, apikey:str, fvecs_file:str = None):
        self._apikey = apikey
        self._cosaccess = CosAccessManager(apikey=self._apikey) # cosaccess simplifies interactions with COS in Python
        self._s3resource = ibm_boto3.resource("s3", ibm_api_key_id=self._apikey,
                                              ibm_service_instance_id=self._cosaccess.get_cos_instance_crn(bucket),
                                              config=Config(signature_version="oauth"),
                                              endpoint_url=self._cosaccess.get_cos_endpoint(bucket))
        self._s3client = ibm_boto3.client("s3", ibm_api_key_id=self._apikey,
                                          ibm_service_instance_id=self._cosaccess.get_cos_instance_crn(bucket),
                                          config=Config(signature_version="oauth"),
                                          endpoint_url=self._cosaccess.get_cos_endpoint(bucket))
        super().__init__(vector_project=vector_project, bucket=bucket, s3endpoint=None, s3accesskey=None, s3secretkey=None, fvecs_file=fvecs_file)
