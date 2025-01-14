#### Jan 13, 2025
Decided to add a logbook to the project.
Fares shared new images, let's try to train a model.
Also Annalisa shared better labeling for same images.

I need a program to separate dataset into train and validation after loading...
Done.

Deleted old bad-performing model for NbSe2 trained with the following features: 1L, 2L, 3L, 4L, 5L, >=5L. (6 classes)

Train new model based on new labeling by Fares of NbSe2 images: 3L, 4L, 5L, >5L. (4 classes)

Noticed that number of classes 6 also works with actual 4 classes... Strange. Changed to 4.

We need to standartize classes and how will we label them.

Trained two times without and with separation for train/val(test). The results are not so good. We need much more data.
Also the last training with validation data took 1,5h... Maybe better to return thinking to transfer it on Franklin?