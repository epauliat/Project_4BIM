import os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")

files = os.listdir('faces')
for name in files:
    print(name)
    picture = os.listdir('faces/'+name)
    for p in picture:
            image = mpimg.imread('faces/'+name+"/"+p)
            plt.title(name)
            plt.imshow(image)
            plt.show()
