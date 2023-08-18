# Copyright 2023 Minttu Alakuijala.
# Copyright 2022 The Scenic Authors.
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

"""setup.py for Human Offline Learned Distances (HOLD), based on the Scenic library.

Install for development:

  pip intall -e . .[testing]
"""

from distutils import cmd
import os
import urllib.request

from setuptools import find_packages
from setuptools import setup
from setuptools.command import install

SIMCLR_DIR = "simclr/tf2"
DATA_UTILS_URL = "https://raw.githubusercontent.com/google-research/simclr/master/tf2/data_util.py"


class DownloadSimCLRAugmentationCommand(cmd.Command):
  """Downloads SimCLR data_utils.py as it's not built into an egg."""
  description = __doc__
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    build_cmd = self.get_finalized_command("build")
    dist_root = os.path.realpath(build_cmd.build_lib)
    output_dir = os.path.join(dist_root, SIMCLR_DIR)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "data_util.py")
    downloader = urllib.request.URLopener()
    downloader.retrieve(DATA_UTILS_URL, output_path)


class InstallCommand(install.install):

  def run(self):
    self.run_command("simclr_download")
    install.install.run(self)


install_requires_projects = [
    "ott-jax>=0.2.0",
    "sklearn",
]

install_requires_core = [
    "absl-py>=1.0.0",
    "dmvr @ git+https://github.com/deepmind/dmvr.git@adf78b7a8edd4bb56c6ce1bffaa5003019761716",
    "numpy>=1.12",
    "jax>=0.2.21,<0.3",
    "jaxlib>=0.1.74,<0.3",
    "flax>=0.4.0,<0.6",
    "ml-collections>=0.1.1",
    "seaborn>=0.11.2",
    "tensorflow<2.8,>=2.7.0",
    "tensorflow-addons>=0.15.0",
    "immutabledict>=2.2.1",
    "clu>=0.0.6",
    "tensorflow-datasets",
    "tfds-nightly>=4.5.2.dev,<5",
    "tf-models-official",
    "tensorflow-probability>=0.15,<0.16",
]

tests_require = [
    "pytest",
] + install_requires_projects

setup(
    name="scenic",
    version="0.0.1",
    description=("Reward model implementation of HOLD, based on the Scenic library."),
    author="Minttu Alakuijala",
    author_email="minttu.alakuijala@aalto.fi",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/minttusofia/hold-rewards",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires_core,
    cmdclass={
        "simclr_download": DownloadSimCLRAugmentationCommand,
        "install": InstallCommand,
    },
    tests_require=tests_require,
    extras_require={
        "testing": tests_require,
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Scenic",
)
