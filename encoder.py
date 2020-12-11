import pickle
import cv2 as cv
import random

# Define the Paths for folders
PATH = "./data/"
PICKLE = "./pickles/"

# Define total number of data for each category
Total = 3700

# Define features and labels arrays
masked = []
not_masked = []


# Make an Enum class
class Enum:
    with_mask = 0
    without_mask = 1


# Iterate through masked images
print("Reading Data.......")
for mask in range(1, Total+1):
    img = cv.imread(PATH + "with_mask/with_mask_" + str(mask) + ".jpg")
    img = cv.resize(img, (100, 100))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    masked.append([img, Enum.with_mask])
    print(mask)

# Iterate through unmasked images
for not_mask in range(1, Total+1):
    img = cv.imread(PATH + "without_mask/without_mask_" + str(not_mask) + ".jpg")
    img = cv.resize(img, (100, 100))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    not_masked.append([img, Enum.without_mask])
    print(not_mask)

# Join the masked and unmasked arrays to make a singular data array
print("Data Collected")
data = masked + not_masked
# Shuffle the data array
random.shuffle(data)

# Divide into test and training data
train = data[:int((Total*2/100)*80)]
test = data[int((Total*2/100)*80):]

# Divide further into X and Y categories
train_X = [x[0] for x in train]
train_Y = [y[1] for y in train]

test_X = [x[0] for x in test]
test_Y = [y[1] for y in test]


print("Dumping...............")
# Finally dumping into pickle!!!
pickle.dump(train_X, open(PICKLE + "X_train.pkl", "wb"))
pickle.dump(train_Y, open(PICKLE + "Y_train.pkl", "wb"))

pickle.dump(test_X, open(PICKLE + "X_test.pkl", "wb"))
pickle.dump(test_Y, open(PICKLE + "Y_test.pkl", "wb"))

print("Done!!!!!!!!!!!!!")

