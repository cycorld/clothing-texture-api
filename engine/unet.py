import cv2
import numpy as np
import tensorflow as tf

# Load U-Net model (pre-trained or your custom trained model)
def load_unet_model(model_path):
    return tf.keras.models.load_model(model_path)

# Texture extraction using U-Net
def extract_texture(image, model):
    input_image = cv2.resize(image, (256, 256)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    mask = model.predict(input_image)[0]  # Predict texture mask
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize back
    mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask
    return mask

# Apply the extracted texture to the product image
def apply_texture(texture_image, product_image, mask):
    texture = cv2.bitwise_and(texture_image, texture_image, mask=mask)
    texture_gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    _, texture_binary = cv2.threshold(texture_gray, 1, 255, cv2.THRESH_BINARY)
    
    product_background = cv2.bitwise_and(product_image, product_image, mask=255-texture_binary)
    combined = cv2.add(product_background, texture)
    return combined

# Main function
if __name__ == "__main__":
    # Load input images
    input_clothing_path = "clothing.jpg"  # Input clothing image
    input_product_path = "product.jpg"    # Product image
    model_path = "unet_model.h5"          # Path to U-Net model
    
    clothing_image = cv2.imread(input_clothing_path)
    product_image = cv2.imread(input_product_path)
    
    # Load pre-trained U-Net model
    unet_model = load_unet_model(model_path)
    
    # Extract texture
    texture_mask = extract_texture(clothing_image, unet_model)
    
    # Apply texture to the product image
    result_image = apply_texture(clothing_image, product_image, texture_mask)
    
    # Save and display the result
    cv2.imwrite("result.jpg", result_image)
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
