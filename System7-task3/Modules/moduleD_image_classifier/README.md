# Module D Practical Task – Develop an Image Classifier

## Goal
Build a multiclass image classifier for CIFAR-10 (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks) using transfer learning, then use it to predict a new image of your own.

## Steps the script performs
1. Load CIFAR-10 and build MobileNetV2 with pretrained ImageNet weights.
2. **Stage 1 – feature extraction:** train only the new classification head, with the base frozen.
3. **Stage 2 – fine-tuning:** unfreeze the top layers of the base and train with a much smaller learning rate.
4. Report test accuracy and per-class accuracy, then save `image_classifier.keras` and a grid of predictions.
5. Optional: predict the class of any image file you provide.

## Run Python Model
```bash
# Train the model and save image_classifier.keras
python Modules/moduleD_image_classifier/image_classifier.py

# Predict a specific photo from terminal
python Modules/moduleD_image_classifier/image_classifier.py --predict path/to/your_photo.jpg
```

## Run Interactive Next.js Frontend UI
An interactive, professional Next.js 14 web application is provided in the `frontend/` folder:

```bash
# Navigate to the frontend directory
cd Modules/moduleD_image_classifier/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```
Open **`http://localhost:3000`** in your browser to drag-and-drop images, test one-click demo samples, and view real-time probability distributions.

## Deliverables
- The printed per-class accuracy and `predictions.png`
- `image_classifier.keras` model file
- Next.js Web App in `frontend/`
- `REPORT.md` in this folder: the test accuracy after each stage, which classes are confused most, and one improvement you tried (for example, more training images or a larger input size).
