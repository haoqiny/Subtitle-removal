import cv2
import json
import boto3
import logging
import ffmpeg
import argparse
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


class Pipeline:

    def __init__(self, readKey: str, specKey: str, saveKey: str):
        self.itemID = Path(readKey).stem
        self.readKey, self.specKey, self.saveKey = readKey, specKey, saveKey
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.framesDir = Path(tempfile.mkdtemp())
        self.tmasksDir = Path(tempfile.mkdtemp())
        self.importDir = Path(tempfile.mkdtemp())
        self.exportDir = Path(tempfile.mkdtemp())
        ## make them local
        self.s3 = None
        self.db = None

    def stageA(self):
        logging.info("StageA: Copy assets")
        self.readPath = Path(self.readKey).resolve()
        self.specPath = Path(self.specKey).resolve()
        try:
            progress_file = Path("/app/output/progress.txt")
            progress_file.write_text("")  # Truncate the file
        except Exception as e:
            logging.error(f"Failed to clear progress.txt: {e}")
            
    def stageB(self):
        logging.info("StageB: Unpack frames")
        self.recordJobStatus("StageB: Unpack frames")
        args = []
        args.extend(["ffmpeg", "-i", self.readPath.as_posix()])
        args.extend([f"{self.framesDir}/%08d.png"])
        subprocess.run(args, check=True)

    def stageC(self):
        logging.info("StageC: Create tmasks")
        self.recordJobStatus("StageC: Create tmasks")
        glob = sorted(self.framesDir.glob("*.png"))
        H, W, _ = cv2.imread(glob[0]).shape
        spec = json.loads(self.specPath.read_text())
        for i, path in enumerate(glob):
            tmask = np.zeros((H, W), dtype=np.uint8)
            for item in spec:
                t0, t1 = item["startAt"], item["endWith"]
                if t0 <= i + 1 <= t1:
                    for rect in item["regions"]:
                        x, y = rect["x"], rect["y"]
                        w, h = rect["w"], rect["h"]
                        tmask[y : y + h, x : x + w] = 255
            cv2.imwrite(f"{self.tmasksDir}/{path.name}", tmask)

    def stageD(self, batchSize: int = 9):
        logging.info("StageD: Batch split")
        self.recordJobStatus("StageD: Batch split")
        framesGlob = sorted(self.framesDir.glob("*.png"))
        tmasksGlob = sorted(self.tmasksDir.glob("*.png"))
        for i, (framePath, tmaskPath) in enumerate(zip(framesGlob, tmasksGlob)):
            assert framePath.name == tmaskPath.name
            folder = Path(self.importDir, f"frames_{i//batchSize+1:08d}")
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            framePath.rename(Path(folder, framePath.name))
            folder = Path(self.importDir, f"tmasks_{i//batchSize+1:08d}")
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmaskPath.rename(Path(folder, tmaskPath.name))

    def getVideoFps(self):
        probeInfo = ffmpeg.probe(self.readPath.as_posix())
        videoInfo = [s for s in probeInfo["streams"] if s["codec_type"] == "video"][0]
        return eval(videoInfo["r_frame_rate"])

    def stageE(self):
        framesGlob = sorted(self.importDir.glob("frames_*"))
        tmasksGlob = sorted(self.importDir.glob("tmasks_*"))
        total_batches = len(framesGlob)
        for i, (frames, tmasks) in enumerate(zip(framesGlob, tmasksGlob)):
            status_msg = f"StageE: Processing batch {i+1}/{total_batches}"
            logging.info(status_msg)
            self.recordJobStatus(status_msg)
            self.writeExternalProgress(status_msg)
            args = []
            args.extend(["conda", "run", "--live-stream", "-n", "pixa"])
            args.extend(["python", "test.py"])
            args.extend(["--model", "e2fgvi_hq"])
            args.extend(["--video", frames.as_posix()])
            args.extend(["--mask", tmasks.as_posix()])
            args.extend(["--ckpt", "release_model/E2FGVI-HQ-CVPR22.pth"])
            args.extend(["--savepath", Path(self.exportDir, f"{i:08d}.mp4").as_posix()])
            args.extend(["--savefps", str(int(self.getVideoFps()))])
            subprocess.run(args, check=True, cwd="E2FGVI")
            with Path(self.exportDir, "concat.txt").open("a") as f:
                f.write(f"file '{Path(self.exportDir, f'{i:08d}.mp4').as_posix()}'\n")

    def stageF(self):
        logging.info("StageF: Merge batches")
        self.recordJobStatus("StageF: Merge batches")
        args = []
        args.extend(["ffmpeg", "-f", "concat", "-safe", "0"])
        args.extend(["-i", Path(self.exportDir, "concat.txt")])
        args.extend(["-c", "copy"])
        args.extend([Path(self.exportDir, "binded.mp4").as_posix()])
        subprocess.run(args, check=True)
        args = []
        args.extend(["ffmpeg", "-i", self.readPath.as_posix()])
        args.extend(["-i", Path(self.exportDir, "binded.mp4").as_posix()])
        args.extend(["-c:v", "copy", "-c:a", "aac"])
        args.extend(["-map", "0:a", "-map", "1:v"])
        self.savePath = Path(self.exportDir, "result.mp4")
        args.extend([self.savePath.as_posix()])
        subprocess.run(args, check=True)

    def stageG(self):
        logging.info("StageG: Save result")
        Path(self.saveKey).write_bytes(self.savePath.read_bytes())

    def writeExternalProgress(self, message: str):
        try:
            with open("/app/output/progress.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
        except Exception as e:
            logging.error(f"Failed to write external progress: {e}")

    def recordJobStatus(self, message: str): pass
    #     try:
    #         self.db.update_item(
    #             Key={"PK": f"ITEM#{self.itemID}", "SK": f"META"},
    #             UpdateExpression="SET JobStatus = :jobStatus",
    #             ExpressionAttributeValues={":jobStatus": message},
    #         )
    #     except Exception as e:
    #         logging.error(e)
    #         raise e

    def recordStartedAt(self): pass
    #     timestamp = datetime.now(timezone.utc).isoformat()
    #     try:
    #         self.db.update_item(
    #             Key={"PK": f"ITEM#{self.itemID}", "SK": f"META"},
    #             UpdateExpression="SET StartedAt = :startedAt",
    #             ExpressionAttributeValues={":startedAt": timestamp},
    #         )
    #     except Exception as e:
    #         logging.error(e)
    #         raise e

    def recordStoppedAt(self, hasFailed: bool = False): pass
    #     timestamp = datetime.now(timezone.utc).isoformat()
    #     try:
    #         self.db.update_item(
    #             Key={"PK": f"ITEM#{self.itemID}", "SK": f"META"},
    #             UpdateExpression="SET StoppedAt = :stoppedAt, HasFailed = :hasFailed",
    #             ExpressionAttributeValues={
    #                 ":stoppedAt": timestamp,
    #                 ":hasFailed": hasFailed,
    #             },
    #         )
    #     except Exception as e:
    #         logging.error(e)
    #         raise e

    def dispatch(self):
        try:
            self.recordStartedAt()
            self.stageA()
            self.stageB()
            self.stageC()
            self.stageD()
            self.stageE()
            self.stageF()
            self.stageG()
            self.recordStoppedAt()
        except Exception as e:
            logging.error(e)
            self.recordStoppedAt(hasFailed=True)
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("readKey", type=str)
    parser.add_argument("specKey", type=str)
    parser.add_argument("saveKey", type=str)
    P = Pipeline(**vars(parser.parse_args()))
    P.dispatch()
