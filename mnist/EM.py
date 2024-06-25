from mnist import Mnist
import numpy as np

k = 3
i,j = 0,1

mnist = Mnist()
images, labels = mnist.image_labels_of_digits([1])

def learn(images, labels):
    pr_label = np.ones((k,1))
    for x in range(k):
        pr_label[x] = sum(labels==x)


    
    pr_image_label = np.ones((k,784))

    for x in range(k):
        pr_image_label[x] += images[labels == x].sum(axis=0)
        pr_image_label[x] /= pr_label[x]

    pr_label /= sum(pr_label)

    return pr_label, pr_image_label


def predict(image, pr_label, pr_image_label):
    log_pr = np.zeros(l)
    for x in range(k):
        log_pr[x] = np.log(pr_label[x]) + ([np.log(pr_image_label[x][p]) if val else np.log(1-pr_image_label[x][p]) for p, val in enumerate(image)])
    


    return np.argmax(log_pr)


def predict_all(images, pr_label, pr_image_label):
    return [predict(image, pr_label, pr_image_label) for image in images]



def em(images):
    N = len(images)
    labels = np.random.randint(0,2,size=(N))
    for _ in range(10):
        pr_label, pr_image_label = learn(images, labels)
        labels = predict_all(images, pr_label, pr_image_label)
    return labels
labels = em(images)
"""
pr_label, pr_image_label = learn(images, labels)

predictions = predict_all(images, pr_label, pr_image_label)
acc = sum(predictions == labels)/len(labels)

print(acc)
"""