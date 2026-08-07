import csv
import shutil
from pathlib import Path

from tqdm import tqdm


projectRoot = Path(__file__).resolve().parents[1]
datasetPath = projectRoot / "data" / "raw" / "firstImpressionsV2"
sampleVideoPath = projectRoot / "data" / "processed" / "firstImpressionsSample" / "videos"
manifestPath = projectRoot / "data" / "processed" / "manifests" / "firstImpressionsSampleManifest.csv"
defaultSampleCount = 25


def findMp4Files():
    if not datasetPath.exists():
        print(f"Dataset folder does not exist yet: {datasetPath}")
        return []

    return sorted(path for path in datasetPath.rglob("*.mp4") if path.is_file())


def splitRank(videoPath):
    pathParts = [part.lower() for part in videoPath.relative_to(datasetPath).parts]

    if "train" in pathParts:
        return 0
    if "validation" in pathParts or "valid" in pathParts or "val" in pathParts:
        return 1
    if "test" in pathParts:
        return 2

    return 3


def chooseSampleVideos(videoFiles, sampleCount):
    sortedVideoFiles = sorted(
        videoFiles,
        key=lambda videoPath: (splitRank(videoPath), str(videoPath.relative_to(datasetPath))),
    )

    return sortedVideoFiles[:sampleCount]


def copySampleVideos(sampleVideos):
    sampleVideoPath.mkdir(parents=True, exist_ok=True)
    manifestPath.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for index, sourcePath in enumerate(tqdm(sampleVideos, desc="Copying sample videos"), start=1):
        sampleId = f"sample{index:03d}"
        sampleFileName = f"{sampleId}_{sourcePath.name}"
        targetPath = sampleVideoPath / sampleFileName

        if not targetPath.exists():
            shutil.copy2(sourcePath, targetPath)

        rows.append(
            {
                "sampleId": sampleId,
                "fileName": sourcePath.name,
                "sourcePath": str(sourcePath.relative_to(projectRoot)),
                "samplePath": str(targetPath.relative_to(projectRoot)),
            }
        )

    with manifestPath.open("w", newline="") as csvFile:
        writer = csv.DictWriter(
            csvFile,
            fieldnames=["sampleId", "fileName", "sourcePath", "samplePath"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def createSample(sampleCount=defaultSampleCount):
    print(f"Looking for .mp4 files in: {datasetPath}")

    videoFiles = findMp4Files()
    print(f"Found {len(videoFiles)} .mp4 files.")

    if not videoFiles:
        print("No .mp4 files found. Run the download script first.")
        return

    sampleVideos = chooseSampleVideos(videoFiles, sampleCount)
    print(f"Creating sample with {len(sampleVideos)} videos.")

    rows = copySampleVideos(sampleVideos)

    print("Sample created.")
    print(f"Sample videos saved to: {sampleVideoPath}")
    print(f"Sample manifest saved to: {manifestPath}")
    print(f"Manifest rows written: {len(rows)}")


if __name__ == "__main__":
    createSample()
