# Module D Practical Task – Develop an Image Classifier

## Goal
Build a multiclass image classifier for CIFAR-10 (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks) using transfer learning, then use it to predict a new image of your own.

## Steps the script performs
1. Load CIFAR-10 and build MobileNetV2 with pretrained ImageNet weights.
2. **Stage 1 – feature extraction:** train only the new classification head, with the base frozen.
3. **Stage 2 – fine-tuning:** unfreeze the top layers of the base and train with a much smaller learning rate.
4. Report test accuracy and per-class accuracy, then save `image_classifier.keras` and a grid of predictions.
5. Optional: predict the class of any image file you provide.

## Run
```bash
python tasks/moduleD_image_classifier/image_classifier.py
python tasks/moduleD_image_classifier/image_classifier.py --predict path/to/your_photo.jpg
```

## Deliverables
- The printed per-class accuracy and `predictions.png`
- `REPORT.md` in this folder: the test accuracy after each stage, which classes are confused most, and one improvement you tried (for example, more training images or a larger input size).
