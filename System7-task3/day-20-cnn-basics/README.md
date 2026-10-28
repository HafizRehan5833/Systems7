# Day 6 – Image Data and Convolutional Neural Networks (Module D)

**Commit:** `day-20-cnn-basics`

## Topics
- Image data and preprocessing: resizing, normalization, augmentation
- Convolution, filters, pooling, and feature maps
- Convolutional Neural Networks (CNNs)

## Practice questions
| # | Level | File | Question |
| --- | --- | --- | --- |
| 16 | Medium | `practice/q16_resize_augment.py` | Resize to 64×64, normalize, augment, display 9 samples |
| 17 | Hard | `practice/q17_cnn_cifar10.py` | CNN with Conv2D and MaxPooling on CIFAR-10: test accuracy and confusion matrix |

The first run downloads CIFAR-10 (about 170 MB). Question 17 takes several minutes on a CPU.

## Learning notes
- **Why CNNs suit images:** What do local filters and weight sharing do that a fully connected layer cannot? Why does a 32×32×3 image need far fewer weights in a convolution layer?
  - My note:

## Completed today
- [ ] Solved Q16 and Q17
- [ ] Wrote the note above
