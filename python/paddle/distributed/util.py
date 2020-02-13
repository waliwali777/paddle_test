# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Cluster():
    def __init__(self):
        self.job_server = None
        self.pods = None

    def __eq__(self, cluster):
        if len(self.pods) != len(cluster.pods):
            return True

        for pod in self.pods:
            if pod != cluster.pods[i]:
                return False

        return True

    def __ne__(self, cluster):
        return not self.__eq__(cluster)

    def update_pods(cluster):
        self.pods = copy.copy(cluster.pods)
        pass

    def world_rank(self):
        count = 0
        for pod in self.pods:
            for gpu in pod.gpus:
                count += 1
        return count

    def get_trainer_endpoints(self):
        pass


class JobServer():
    def __init__(self):
        self.endpoint = None


class Trainer():
    def __init__(self):
        self.gpu = []
        self.endpoint = []
        self.rank = []

    def __eq__(self):
        pass

    def __ne__(self):
        pass

    def ranks(self):
        return self.rank


class Pod():
    def __init__(self):
        self.idx = None
        self.ip = None
        self.port = None

        self.trainers = []

    def __eq__(self, pod):
        pass

    def __ne__(self, pod):
        return not self == pod

    def parse_response(self, res_pods):
        pass

    def ranks(self):
        r = []
        for t in self.trainers:
            r.extend(t.ranks)

        return r
