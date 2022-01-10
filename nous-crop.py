import pandas as pd
import cv2

#crops = pd.read_csv("data/input/csv_annotations_short.csv", header=0, usecols=['filename', 'x1', 'y1', 'x2', 'y2'])

annotationCsvFile = "/NAS/BeeNas/VHL_Algemeen/Projecten/006_plakplaten/ai/NOUSoutput/1200dpi_insect_detection_v0.8/csv_annotations_2.csv"
#"data/input/csv_annotations_short.csv"
inputFolder = "/NAS/BeeNas/VHL_Algemeen/Projecten/006_plakplaten/ai/01_Nous-insect-train_all_st_2021-11/all_scans/"
# "data/input/"
outputFolder = "/NAS/BeeNas/VHL_Algemeen/Projecten/006_plakplaten/ai/01_Nous-insect-train_all_st_2021-11/annotated_insect_crops_2022-01-09_2/"
#"data/output/"
extractLabel = True
extractScore = False

# read the NOUS annotation csv output file into a dataframe
crops = pd.read_csv(annotationCsvFile, header=0, usecols=['filename', 'x1', 'y1', 'x2', 'y2', 'score', 'label'])
# relevant column labels:
# - filename, x1, y1, x2, y2

for fName, x1, y1, x2, y2, score, label in zip(crops['filename'], crops['x1'], crops['y1'], crops['x2'], crops['y2']):
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    print(fName, x1, y1, x2, y2, score, label)
    img = cv2.imread(inputFolder+fName+".jpg")
    print(img.shape)
    # check if the crop boundaries are outside the picture and correct
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > img.shape[1]:
        x2 = img.shape[1]
    if y2 > img.shape[0]:
        y2 = img.shape[0]

    cropFileName = fName + '_' + str(x1) + ':' + str(y1) + ':' + str(x2) + ':' + str(y2)
    if extractScore:
        cropFileName = cropFileName + '_' + score
    if extractLabel:
        cropFileName = cropFileName + '_' + label
    cropFileName = cropFileName + '.jpg'
    crop = img[y1:y2, x1:x2]
    print(crop.shape)
    cv2.imwrite(outputFolder+cropFileName, crop)
    #cv2.imshow('cropped', crop)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
