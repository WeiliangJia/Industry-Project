import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import nibabel as nib
import numpy as np
from YOLO import OD
from YOLO import predict_tumor

class ImageUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tumor Detection")

        # Create main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(padx=40, pady=40)

        # Layout configuration
        self.frames = []
        self.image_labels = []
        self.label_texts = [
            ("upload", "image detection"),
            ("tumor detection", "true / false"),
            ("object detection", "scale"),
            ("mask", ""),
            ("tumor observation", "size: boundary"),

        ]

        # Create frames and assign functions to buttons
        for i, (button_text, label_text) in enumerate(self.label_texts):
            frame = tk.Frame(self.main_frame, borderwidth=2, relief="groove", width=256, height=256)
            frame.grid(row=i//3, column=i%3, padx=15, pady=15)
            frame.pack_propagate(0)  # Disable frame auto-resizing

            # Assign upload function only to the first button
            if i == 0:
                button = tk.Button(frame, text=button_text, command=self.upload_image)
            else:
                # Assign other functions to the remaining buttons
                #后面加功能在这加###
                button = tk.Button(frame, text=button_text, command=lambda l=i: self.OD_predict(l))

                button = tk.Button(frame, text=button_text, command=lambda l=i: self.other_function(l))

            button.pack(side="top", pady=5)

            image_label = tk.Label(frame)
            image_label.pack(expand=True)

            label = tk.Label(frame, text=label_text, fg="gray")
            label.pack(side="bottom", pady=5)

            self.frames.append(frame)
            self.image_labels.append(image_label)

        # Store image paths
        self.image_paths = [None] * len(self.label_texts)  # Adjusted size to match the number of buttons

    def upload_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(filetypes=[("Analyze/NIfTI files", "*.hdr *.img *.nii")])
        if file_path:
            self.image_paths[0] = file_path  # Only store path for the first image
            # Load the image using nibabel
            img = nib.load(file_path)
            img_data = img.get_fdata()
            # Debugging: print the shape and dtype of the image data
            print(f"Data shape: {img_data.shape}, dtype: {img_data.dtype}")

            # Handle 3D or 4D data by selecting a 2D slice
            if img_data.ndim == 3:
                slice_2d = img_data[:, :, img_data.shape[2] // 2]
            elif img_data.ndim == 4:
                slice_2d = img_data[:, :, img_data.shape[2] // 2, 0]
            else:
                messagebox.showerror("Error", "Unsupported image dimensions.")
                return

            # Normalize the slice data for display
            slice_2d_normalized = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d)) * 255
            slice_2d_normalized = slice_2d_normalized.astype(np.uint8)

            # Convert to PIL Image for display
            img_pil = Image.fromarray(slice_2d_normalized)
            #img_pil = img_pil.resize((128, 128), Image.ANTIALIAS)

            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_labels[0].config(image=img_tk)
            self.image_labels[0].image = img_tk
            messagebox.showinfo("Success", f"Image uploaded: {file_path}")

    def other_function(self, index):
        # Example: Print a message or perform any other action
        messagebox.showinfo("Action", f"Button {index + 1} pressed. Performing its unique function.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageUploaderApp(root)
    root.mainloop()
