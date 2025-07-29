import os
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def custom_loss(y_true, y_pred):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = tf.reduce_mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

model = tf.keras.models.load_model(
    r"c:\Users\starw\OneDrive\Documents\deeplab\deeplabv3plus_model_IOU_2.h5",
    custom_objects={'custom_loss': custom_loss, 'iou_metric': iou_metric}
)

image_directory = "./data_annotated/test_json/MRI/JPEGImages"
ground_truth_directory = "./data_annotated/test_json/MRI/SegmentationClassPNG3"
output_directory = "./data_annotated/test_MRI_output_images"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

iou_values = []

for image_file in os.listdir(image_directory):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(image_directory, image_file)
        ground_truth_path = os.path.join(ground_truth_directory, image_file.replace(".jpg", ".png"))

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((512, 512))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        output = model(image)
        predicted_mask = np.argmax(output[0], axis=-1)

        # Scale up the predicted mask by multiplying with 255.0
        predicted_mask_scaled = (predicted_mask * 255).astype(np.uint8)

        # Reshape the predicted mask to 3 channels
        predicted_mask_rgb = np.zeros_like(image[0], dtype=np.uint8)
        predicted_mask_rgb[:, :, 0] = predicted_mask_scaled
        predicted_mask_rgb[:, :, 1] = predicted_mask_scaled
        predicted_mask_rgb[:, :, 2] = predicted_mask_scaled

        ground_truth = Image.open(ground_truth_path)
        ground_truth = ground_truth.resize((512, 512))
        ground_truth = (np.array(ground_truth) > 0.5).astype(np.uint8)

        # Save the reshaped output image
        output_image = Image.fromarray(predicted_mask_rgb)
        output_image_path = os.path.join(output_directory, f"{image_file.replace('.jpg', '_output.jpg')}")
        output_image.save(output_image_path)

        intersection = np.logical_and(predicted_mask, ground_truth)
        union = np.logical_or(predicted_mask, ground_truth)

        if np.sum(union) == 0:
            iou = 0.0
        else:
            iou = np.sum(intersection) / np.sum(union)
        
        iou_values.append(iou)

        print(f"IoU for {image_file}: {iou:.3f}")

# Plotting IoU values
plt.bar(range(len(iou_values)), iou_values, tick_label=os.listdir(image_directory))
plt.xlabel('Image')
plt.ylabel('IoU Value')
plt.title('IoU Values for Test Images')
plt.show()