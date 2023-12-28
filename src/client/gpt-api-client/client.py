"""
@brief: A Chat-GPT response generating client using Async io
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: March 19, 2023
"""

import os.path
import random
import string
import time
import json
import openai
import asyncio

from enum import Enum
from typing import Dict, Any, List, Set
from pathlib import Path

class RequestResult(Enum):
    SUCCESS = 1     # Success
    RETRY   = 0     # Failed, but can retry. e.g. Blocked by API Threshold
    INVALID = -1    # No need to retry. e.g. Blocked by OpenAI content policy
    PENDING = -2    # Not requested yet


class OpenGPTTextAsyncClient:
    # Client Configurations
    MAX_ASYNC_WORKER_COUNT = 20
    MAX_TOKEN_COST = float("inf")
    MAX_LENGTH_ALLOWED = 2e3
    TOKEN_SPLITTER = {p for collection in [string.punctuation, string.whitespace] for p in collection}
    # How we handle different request result?
    HANDLE_STRATEGY = {
        "stop"          : RequestResult.SUCCESS,
        "length"        : RequestResult.RETRY,      # Maybe INVALID ? - response length exceeds the limit
        "content_filter": RequestResult.INVALID,
        "null"          : RequestResult.RETRY,
        # Additional handles
        "unknown"       : RequestResult.RETRY,
        "rate_limit"    : RequestResult.RETRY,
        "length_limit"  : RequestResult.INVALID,    # Request length exceeds the limit
        "invalid_char"  : RequestResult.INVALID
    }
    CANCELED = False

    @staticmethod
    def _must_exist(dict: Dict[str, Any], key: str):
        if key not in dict: raise Exception(f"Missing `{key}` key")

    @staticmethod
    def _load_client_state() -> Dict[str, Any]:
        if not os.path.exists("./src/client/gpt-api-client/client_state.json"):
            return {"tokens": 0, "retry": [], "invalid": []}
        with open("./src/client/gpt-api-client/client_state.json", "r") as f:
            return json.load(f)

    def _dump_client_state(self):
        with open("./src/client/gpt-api-client/client_state.json", "w+") as f:
            json.dump({
                "tokens" : self.tokens,
                "retry"  : self.retry,
                "invalid": list(self.invalid)
            }, f, indent="\t")

    @staticmethod
    def _estimate_token_count(sample: str) -> int:
        est_num = 0
        for char in sample:
            est_num += 1 if char in OpenGPTTextAsyncClient.TOKEN_SPLITTER else 0
        return est_num

    def _sample_with_existing(self, allHumanEntries: Dict[str, str], existEntries: Set[str]) -> Set[str]:
        """
        Given a set of all human text files, return a set of UIDs that can does not exist in "existEntries" and
        we have total number of enties + exist entry = len(allHuman) * self.sample_factor
        """
        target_cnt = int(len(allHumanEntries) * self.sampling_factor)
        exist_cnt = len(existEntries)
        remain_cnt = max(target_cnt - exist_cnt, 0)
        remain_uids = random.choices(list(allHumanEntries.keys()), k=remain_cnt)
        return set(remain_uids)


    def __init__(self, config: Dict[str, str], apiKey: str, sampling: float) -> None:
        self.config = config
        self.sampling_factor = sampling

        # Check if the configuration file is valid
        self._must_exist(self.config, "InputDirectory")
        self._must_exist(self.config, "OutputDirectory")
        self._must_exist(self.config, "InputSubsets")
        self._must_exist(self.config, "ResponseModel")
        self._must_exist(self.config, "QuestionPrompt")

        self.folder = Path(self.config["OutputDirectory"])
        self.folder.mkdir(parents=True, exist_ok=True)
        self.subsets = self.config["InputSubsets"]
        for subset in self.subsets:
            Path(self.folder, subset + ".jsonl").touch(exist_ok=True)

        self.model = self.config["ResponseModel"]
        self.prompt = self.config["QuestionPrompt"]

        # Load internal states
        internal_state = self._load_client_state()
        self.tokens : int                   = internal_state["tokens"]
        self.invalid: Set[str]  = set(uid for uid in internal_state["invalid"])
        self.retry  : List[str] = [uid for uid in internal_state["retry"]]

        # Stop overwhelming openai API to avoid over-threshold request rate
        self.worker_lock = None
        self.writer_lock = None
        self.finish_task = 0
        self.total_task  = 0

        openai.api_key = apiKey

    async def rephrase_request(self, subset: str, humanTextUID: str, humanText: str, wait_time: float = 60.0) -> RequestResult:
        # If there are too many workers, wait until one "retires"
        await self.worker_lock.acquire()

        if self.tokens >= self.MAX_TOKEN_COST: raise Exception("Exceed the MAX_TOKEN_COST setting in GPTTextAsyncClient")

        start_time = time.time()

        result = RequestResult.PENDING
        # Ready ... now Work!

        estimatedNumTokens = self._estimate_token_count(humanText)
        if estimatedNumTokens > self.MAX_LENGTH_ALLOWED:
            print("[x]\t", humanTextUID, "failed since it exceeds the token limit (" + str(self.MAX_LENGTH_ALLOWED) + ")")
            self.worker_lock.release()
            return self.HANDLE_STRATEGY["length_limit"]

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.prompt + "\n\n" + humanText}
                ]
            )

        except openai.error.InvalidRequestError:
            # no need to wait, since the request is not sent for some reason
            await asyncio.sleep(1.0)    # Avoid flushing the API
            self.worker_lock.release()
            return self.HANDLE_STRATEGY["unknown"]
        
        except openai.error.RateLimitError:
            # has to wait
            await asyncio.sleep(wait_time)
            self.worker_lock.release()
            return self.HANDLE_STRATEGY["rate_limit"]

        except (openai.error.APIError, openai.error.TryAgain, openai.error.Timeout):
            await asyncio.sleep(wait_time)
            self.worker_lock.release()
            return self.HANDLE_STRATEGY["unknown"]

        finishReason = response["choices"][0]["finish_reason"]
        result = self.HANDLE_STRATEGY[finishReason]

        if result == RequestResult.SUCCESS:
            machineText = response["choices"][0]["message"]["content"].strip()

            await self.writer_lock.acquire()
            with open(Path(config["OutputDirectory"], subset + ".jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "uid": humanTextUID,
                    "text": machineText
                }))
                f.write("\n")
            self.writer_lock.release()

        self.tokens += response["usage"]["total_tokens"]

        # Wait for 60 secs, then release the lock to spawn a new worker coroutine
        # (We won't be blocked out)
        end_time = time.time()
        await asyncio.sleep(wait_time - (end_time - start_time))
        self.worker_lock.release()

        return result

    async def rephrase_worker(self, subset: str, humanTextUID: str, humanText: str, max_retry: int = 3):
        if humanTextUID in self.invalid:
            self.finish_task += 1
            return
        try:
            request_state = RequestResult.PENDING
            for _ in range(max_retry):
                request_state = await self.rephrase_request(subset, humanTextUID, humanText)
                if request_state == RequestResult.SUCCESS or request_state == RequestResult.INVALID: break
        except KeyboardInterrupt as e:
            self.retry.append(humanTextUID)
            raise e
        except asyncio.CancelledError as e:
            if not self.CANCELED:
                print("All tasks canceled by asyncio")
                self._dump_client_state()
                self.CANCELED = True
            raise e

        self.finish_task += 1
        if request_state == RequestResult.INVALID:
            print("\t",     f"[{self.finish_task} / {self.total_task}]\t", humanTextUID, "\t- INVALID")
            self.invalid.add(humanTextUID)
        elif request_state == RequestResult.RETRY:
            print("\t",     f"[{self.finish_task} / {self.total_task}]\t", humanTextUID, "\t- RETRY")
            self.retry.append(humanTextUID)
        else:
            print("=>\t",   f"[{self.finish_task} / {self.total_task}]\t", humanTextUID, "\t- SUCCESS")

    async def rephrase_dispatcher(self):
        print("Creating tasks ...")
        # Create a lock in the async process to avoid "future from different run loop"
        self.worker_lock = asyncio.Semaphore(OpenGPTTextAsyncClient.MAX_ASYNC_WORKER_COUNT)
        self.writer_lock = asyncio.Semaphore(1) # A mutex lock for writing

        try:
            for subset in self.subsets:
                print("Processing",  subset)
                TASKS = []

                # Get human text entries
                humanTextEntries = dict()
                with open(Path(config["InputDirectory"], subset+".jsonl"), "r") as f:
                    lines = f.read().strip().split("\n")
                    for line in lines:
                        entry = json.loads(line)
                        humanTextEntries[entry["uid"]] = entry["text"]

                # convertedFiles = set(file.name for file in Path(self.folder, subset).iterdir())
                convertedEntries = set()
                with open(Path(config["OutputDirectory"], subset+".jsonl"), "r") as f:
                    lines = f.read().strip().split("\n")
                    for line in lines:
                        if line == "": continue
                        entry = json.loads(line)
                        convertedEntries.add(entry["uid"])

                sampledUIDs = self._sample_with_existing(humanTextEntries, convertedEntries)

                for humanTextUID in sampledUIDs:
                    is_converted = humanTextUID in convertedEntries
                    is_invalid   = humanTextUID in self.invalid

                    if (not is_converted) and (not is_invalid):
                        task = asyncio.create_task(self.rephrase_worker(subset, humanTextUID, humanTextEntries[humanTextUID]))
                        TASKS.append(task)


                self.finish_task = 0
                self.total_task = len(TASKS)
                print(f"Subset Tasks created, {self.total_task} pending")
                await asyncio.gather(*TASKS)
                print("Subset Tasks compete")
            
            print("All Tasks Complete.")
        except Exception as e:
            print("Client interrupted by exception, saving internal state ...")
            print("Exit.")
            raise e
        finally:
            self._dump_client_state()


if __name__ == "__main__":
    CLI_PATH = Path("./src/client", "gpt-api-client")

    secret_path = Path(CLI_PATH, "secret.json")
    config_path = Path(CLI_PATH, "config.json")

    if not os.path.exists(secret_path):
        API_KEY = input("Didn't find OpenAI API key, input it here (a secret file will be created automatically)\n"
                        "OpenAI Secret Key:")
        with open(secret_path, "w") as f:
            json.dump({"OPENAI_API_KEY": API_KEY}, f, indent="\t")

    config = json.load(open(config_path))
    secret = json.load(open(secret_path))
    client = OpenGPTTextAsyncClient(config, secret["OPENAI_API_KEY"], sampling=0.01)
    asyncio.run(client.rephrase_dispatcher())
