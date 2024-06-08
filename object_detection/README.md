# object detection

This model detects furnitures in images.

## Usage

✋ Note for M1 Mac users: This model uses TensorFlow, which does not currently work on M1 machines using Docker. See [replicate/cog#336](https://github.com/replicate/cog/issues/336) for more information.

---

First, make sure you've got the [latest version of Cog](https://github.com/replicate/cog#install) installed.

The pre-trained model is stored in this repo.

Build the image:

```sh
cog build
```

Now you can run predictions on the model:

```sh
cog predict -i image=@cat.png
```