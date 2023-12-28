"""
@brief: A async client used to collect OpenAI's GPT-classifier's response
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: May 4th, 2023
"""
import time
import json
import asyncio

from request_func import zerogpt
from enum import Enum
from typing import Dict, Any
from pathlib import Path

class RequestResult(Enum):
    SUCCESS = 1     # Success
    RETRY   = 0     # Failed, but can retry. e.g. Blocked by API Threshold
    INVALID = -1    # No need to retry. e.g. Blocked by OpenAI content policy
    PENDING = -2    # Not requested yet


class ZeroGPTClassifierClient:
    # Client Configurations
    MAX_ASYNC_WORKER_COUNT = 60
    CANCELED = False

    @staticmethod
    def _must_exist(dict: Dict[str, Any], key: str):
        if key not in dict: raise Exception(f"Missing `{key}` key")

    def __init__(self, config: Dict[str, str]) -> None:
        self.config = config

        # Check if the configuration file is valid
        self._must_exist(self.config, "InputDirectory")
        self._must_exist(self.config, "OutputDirectory")
        self._must_exist(self.config, "InputFile")
        self._must_exist(self.config, "OutputFile")

        self.folder = Path(self.config["OutputDirectory"])
        self.folder.mkdir(parents=True, exist_ok=True)

        # Stop overwhelming openai API to avoid over-threshold request rate
        self.worker_lock = None
        self.writer_lock = None
        self.finish_task = 0
        self.total_task  = 0

    async def rephrase_request(self, subset: str, humanTextUID: str, humanText: str, wait_time: float = 60.0) -> RequestResult:
        # If there are too many workers, wait until one "retires"
        await self.worker_lock.acquire()
        start_time = time.time()

        # Ready ... now Work!
        response, status_code = zerogpt(humanTextUID, humanText)
        
        if status_code != 200:
            await asyncio.sleep(wait_time)
            self.worker_lock.release()
            return RequestResult.RETRY

        await self.writer_lock.acquire()
        with open(Path(config["OutputDirectory"], subset), "a", encoding="utf-8") as f:
            f.write(json.dumps(response))
            f.write("\n")
        self.writer_lock.release()

        # Wait for 60 secs, then release the lock to spawn a new worker coroutine
        # (We won't be blocked out)
        end_time = time.time()
        await asyncio.sleep(wait_time - (end_time - start_time))
        self.worker_lock.release()
        return RequestResult.SUCCESS

    async def rephrase_worker(self, subset: str, humanTextUID: str, humanText: str, max_retry: int = 3):
        try:
            for retry_cnt in range(max_retry):
                result = await self.rephrase_request(subset, humanTextUID, humanText)
                if result == RequestResult.SUCCESS: break
                print(f"{humanTextUID} - Retrying ({retry_cnt + 1} / {max_retry})")
            
        except asyncio.CancelledError as e:
            if not self.CANCELED:
                print("All tasks canceled by asyncio")
                self.CANCELED = True
            raise e
        
        self.finish_task += 1
        if result == RequestResult.SUCCESS:
            print("\t",     f"[{self.finish_task} / {self.total_task}]\t", humanTextUID, "\t- SUCCESS")
        else:
            print("\t",     f"[{self.finish_task} / {self.total_task}]\t", humanTextUID, "\t- FAILED")

    async def rephrase_dispatcher(self):
        print("Creating tasks ...")
        # Create a lock in the async process to avoid "future from different run loop"
        self.worker_lock = asyncio.Semaphore(ZeroGPTClassifierClient.MAX_ASYNC_WORKER_COUNT)
        self.writer_lock = asyncio.Semaphore(1) # A mutex lock for writing

        # Read the existing results
        output_path = Path(self.config["OutputDirectory"], self.config["OutputFile"])
        existing_uids = set()
        if output_path.exists():
            with open(output_path, "r") as f:
                lines = f.read().strip().split("\n")
            for line in lines:
                existing_uids.add(json.loads(line)["uid"])

        try:
            TASKS = []
            with open(Path(self.config["InputDirectory"], self.config["InputFile"]), "r") as f:
                lines = f.read().strip().split("\n")
                for line in lines:
                    if line == "": continue
                    entry = json.loads(line)
                    if entry[UID_KEY] in existing_uids: continue

                    task = asyncio.create_task(self.rephrase_worker(self.config["OutputFile"], entry[UID_KEY], entry["text"]))
                    TASKS.append(task)

            self.finish_task = 0
            self.total_task = len(TASKS)
            print(f"Tasks created, {self.total_task} pending")
            await asyncio.gather(*TASKS)
            
            print("All Tasks Complete.")
        except Exception as e:
            print("Client interrupted by exception, saving internal state ...")
            print("Exit.")
            raise e


if __name__ == "__main__":
    # Since GPT2-output has different format comparing with OpenGPTText
    # We need this to refer to the key of UID for each entry in the dataset
    UID_KEY = "id"
    CLI_PATH = Path("src", "baseline")

    config_path = Path(CLI_PATH, "zerogpt_config.json")
    assert config_path.exists()

    config = json.load(cfg_handle := open(config_path))
    cfg_handle.close()

    client = ZeroGPTClassifierClient(config)
    asyncio.run(client.rephrase_dispatcher())
