from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
datasetPath = projectRoot / "data" / "raw" / "firstImpressionsV2"
videoExtensions = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}


def printFolderStructure(maxDepth=3):
    print("Folder structure:")

    if not datasetPath.exists():
        print(f"Dataset folder does not exist yet: {datasetPath}")
        return

    for path in sorted(datasetPath.rglob("*")):
        relativePath = path.relative_to(datasetPath)

        if len(relativePath.parts) > maxDepth:
            continue

        indent = "  " * (len(relativePath.parts) - 1)
        marker = "/" if path.is_dir() else ""
        print(f"{indent}{path.name}{marker}")


def findVideoFiles():
    if not datasetPath.exists():
        return []

    return sorted(
        path
        for path in datasetPath.rglob("*")
        if path.is_file() and path.suffix.lower() in videoExtensions
    )


def inspectDataset():
    print(f"Inspecting dataset folder: {datasetPath}\n")

    printFolderStructure()

    videoFiles = findVideoFiles()
    print(f"\nVideo files found: {len(videoFiles)}")

    if videoFiles:
        print("\nExample video files:")
        for videoPath in videoFiles[:10]:
            print(videoPath.relative_to(projectRoot))
    else:
        print("\nNo common video files were found yet.")


if __name__ == "__main__":
    inspectDataset()
