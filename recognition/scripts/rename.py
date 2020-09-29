import os
import sys
path = "/home/arpit/Projects/Survelliance_System/data/extra2/"
i = 460
for file1 in os.listdir(path):
    os.rename(os.path.join(path, file1), os.path.join(
        path, "subject03" + "." + str(i) + ".jpg"))
    i = i + 1
