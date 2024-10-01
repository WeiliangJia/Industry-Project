from unet_model import unet_model, compile_and_train_unet, save_unet_model

# Assuming the images and masks have already been preprocessed
# images, masks = load_data()  # Load your image and mask data here

# Create the U-Net model
model = unet_model()

# Compile and train the U-Net model
history = compile_and_train_unet(model, images, masks, epochs=50, batch_size=16)

# Save the trained model
save_unet_model(model, 'unet_brain_tumor.h5')
