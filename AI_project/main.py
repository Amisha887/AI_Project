import numpy as np
import cv2

prototxt=("./models/colorization_deploy_v2.prototxt")
points=("./models/pts_in_hull.npy")
model=("./models/colorization_release_v2.caffemodel")
print("Loading the provided models....")
layer = cv2.dnn.readNetFromCaffe(prototxt,model)
pts = np.load(points)

class8 = layer.getLayerId("class8_ab")
conv8 = layer.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
layer.getLayer(class8).blobs = [pts.astype("float32")]
layer.getLayer(conv8).blobs = [np.full([1, 313], 2, dtype="float32")]

image = cv2.imread('./image/flower.jpg')
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

print("Colorizing the image...")
layer.setInput(cv2.dnn.blobFromImage(L))
ab = layer.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original image", image)
cv2.imshow("Colorized output image", colorized)
cv2.waitKey(0)