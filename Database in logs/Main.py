import cv2
import tkinter as tk
from tkinter import Button
from tkinter import messagebox
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime

class CameraApp:
    def __init__(self, root, database_name):
        self.root = root
        self.root.title("Camera App")
        self.database_name = database_name
        self.capture = cv2.VideoCapture(0)
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root)
        self.label.pack(padx=10, pady=10)

        self.btn_store = Button(self.root, text="Store", command=self.store_image)
        self.btn_store.pack(pady=10)

        self.btn_exit = Button(self.root, text="Exit", command=self.exit_app)
        self.btn_exit.pack(pady=10)

        self.update_camera()

    def update_camera(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.label.imgtk = img
            self.label.configure(image=img)
            self.label.after(10, self.update_camera)

import cv2
import tkinter as tk
from tkinter import Button
from tkinter import messagebox
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime

class CameraApp:
    def __init__(self, root, database_name):
        self.root = root
        self.root.title("Camera App")
        self.database_name = database_name
        self.capture = cv2.VideoCapture(0)
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root)
        self.label.pack(padx=10, pady=10)

        self.btn_store = Button(self.root, text="Store", command=self.store_image)
        self.btn_store.pack(pady=10)

        self.btn_exit = Button(self.root, text="Exit", command=self.exit_app)
        self.btn_exit.pack(pady=10)

        self.update_camera()

    def update_camera(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.label.imgtk = img
            self.label.configure(image=img)
            self.label.after(10, self.update_camera)

    def store_image(self):
        ret, frame = self.capture.read()
        if ret:
            image_name = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            cv2.imwrite(image_name, frame)  # Save the image using OpenCV without color conversion
            self.save_to_database(image_name)
            messagebox.showinfo("Success", f"Image '{image_name}' stored in the database!")


    def save_to_database(self, image_name):
        try:
            conn = sqlite3.connect(self.database_name)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, name TEXT)")
            cursor.execute("INSERT INTO images (name) VALUES (?)", (image_name,))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error: {e}")
        finally:
            if conn:
                conn.close()

    def exit_app(self):
        self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    db_name = "image_database.db"
    root = tk.Tk()
    app = CameraApp(root, db_name)
    root.mainloop()


    def save_to_database(self, image_name):
        try:
            conn = sqlite3.connect(self.database_name)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, name TEXT)")
            cursor.execute("INSERT INTO images (name) VALUES (?)", (image_name,))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error: {e}")
        finally:
            if conn:
                conn.close()

    def exit_app(self):
        self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    db_name = "image_database.db"
    root = tk.Tk()
    app = CameraApp(root, db_name)
    root.mainloop()
