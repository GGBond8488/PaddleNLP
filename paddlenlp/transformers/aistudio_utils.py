# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from aistudio_sdk.hub import download


class UnauthorizedError(Exception):
    pass


class EntryNotFoundError(Exception):
    pass


def aistudio_download(repo_id: str, filename: str):
    # TODO: add arguments such as cache_dir, revision, etc.
    res = download(
        repo_id=repo_id,
        filename=filename,
    )
    if "path" in res:
        return res["path"]
    else:
        if res["error_code"] == 10001:
            raise ValueError("Illegal argument error")
        elif res["error_code"] == 10002:
            raise UnauthorizedError(
                "Unauthorized Access. Please ensure that you have provided the AIStudio Access Token and you have access to the requested asset"
            )
        elif res["error_code"] == 12001:
            raise EntryNotFoundError(f"Cannot find the requested file '{filename}' in repo '{repo_id}'")
        else:
            raise Exception(f"Unknown error: {res}")
