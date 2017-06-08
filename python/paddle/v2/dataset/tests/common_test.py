# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import paddle.v2.dataset.common
import unittest
import tempfile
import glob


class TestCommon(unittest.TestCase):
    def test_md5file(self):
        _, temp_path = tempfile.mkstemp()
        with open(temp_path, 'w') as f:
            f.write("Hello\n")
        self.assertEqual('09f7e02f1290be211da707a266f153b3',
                         paddle.v2.dataset.common.md5file(temp_path))

    def test_download(self):
        yi_avatar = 'https://avatars0.githubusercontent.com/u/1548775?v=3&s=460'
        self.assertEqual(
            paddle.v2.dataset.common.DATA_HOME + '/test/1548775?v=3&s=460',
            paddle.v2.dataset.common.download(
                yi_avatar, 'test', 'f75287202d6622414c706c36c16f8e0d'))

    def test_split(self):
        def test_reader():
            def reader():
                for x in xrange(10):
                    yield x

            return reader

        _, temp_path = tempfile.mkstemp()
        paddle.v2.dataset.common.split(
            test_reader(), 4, suffix=temp_path + '/test-%05d.pickle')
        files = glob.glob(temp_path + '/test-%05d.pickle')
        self.assertEqual(len(files), 3)

    def test_cluster_file_reader(self):
        _, temp_path = tempfile.mkstemp()
        for x in xrange(5):
            with open(temp_path + '/%05d.test' % x) as f:
                f.write('%d\n' % x)
        reader = paddle.v2.dataset.common.cluster_files_reader(
            temp_path + '/*.test', 5, 0)
        for idx, e in enumerate(reader()):
            self.assertEqual(e, str("0"))

    def test_convert(self):
        def test_reader():
            def reader():
                for x in xrange(10):
                    yield x

            return reader

        path = tempfile.mkdtemp()

        paddle.v2.dataset.common.convert(path,
                                         test_reader(), 4, 'random_images')

        files = glob.glob(temp_path + '/random_images-*')
        self.assertEqual(len(files), 3)


if __name__ == '__main__':
    unittest.main()
