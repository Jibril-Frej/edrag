import os

for files in os.listdir("data/AICC_2023/documents"):
    if files.endswith(".tex"):
        with open(f"data/AICC_2023/documents/{files}", "r") as f:
            print(files)
            exercices = f.read()
            print(exercices)
            exercices = exercices.split("\problem")
            for exercice in exercices:
                print("**********")
                print(exercice.rstrip())
                print("**********")
            break
