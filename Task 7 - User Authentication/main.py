import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import Button, messagebox, Label, StringVar, OptionMenu, Entry
from PIL import Image, ImageTk
import sqlite3
from datetime import datetime
from ultralytics import YOLO

def create_credentials_db():
    conn = sqlite3.connect("credentials.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS super_admins (
                      id INTEGER PRIMARY KEY,
                      username TEXT UNIQUE,
                      password TEXT)''')
    cursor.execute("INSERT OR IGNORE INTO super_admins (username, password) VALUES (?, ?)",
                   ("admin", "password"))
    conn.commit()
    conn.close()

create_credentials_db()

class CameraApp:
    def __init__(self, root, database_name, user_role):
        self.root = root
        self.root.title("Privacy Enhancement")
        self.root.geometry("1200x800")
        self.database_name = database_name
        self.capture = cv2.VideoCapture(0)
        self.current_frame = None
        self.background = None
        self.privacy_mode = False
        self.user_role = user_role
        self.model = YOLO('yolov8n-seg.pt')
        self.inpainting_done = False

        self.create_widgets()
        self.update_camera()

    def create_widgets(self):
        try:
            bg_image = Image.open('Task 6 - GUI and database integration\AI.jpg')
            bg_image = bg_image.resize((1200, 800), Image.LANCZOS)
            bg_photo = ImageTk.PhotoImage(bg_image)

            bg_label = tk.Label(self.root, image=bg_photo)
            bg_label.place(relwidth=1, relheight=1)
            bg_label.image = bg_photo
        except Exception as e:
            print(f"Error loading background image: {e}")

        main_frame = tk.Frame(self.root, bg="#f7f5f2")
        main_frame.place(relx=0.5, rely=0.4, anchor="center")

        if self.user_role != "User":
            self.before_frame = tk.Frame(main_frame, width=500, height=500, borderwidth=2, relief="solid", bg="#ffffff")
            self.before_frame.grid(row=0, column=0, padx=20, pady=10)
            self.before_canvas = tk.Canvas(self.before_frame, width=500, height=500)
            self.before_canvas.pack()
            before_label = tk.Label(main_frame, text="Before privacy enhancement", font=("Helvetica", 14), bg="#f7f5f2", fg="#a83232")
            before_label.grid(row=1, column=0, pady=10)

        if self.user_role != "Admin":
            self.after_frame = tk.Frame(main_frame, width=500, height=500, borderwidth=2, relief="solid", bg="#ffffff")
            self.after_frame.grid(row=0, column=1, padx=20, pady=10)
            self.after_canvas = tk.Canvas(self.after_frame, width=500, height=500)
            self.after_canvas.pack()
            after_label = tk.Label(main_frame, text="After privacy enhancement", font=("Helvetica", 14), bg="#f7f5f2", fg="#a83232")
            after_label.grid(row=1, column=1, pady=10)

        button_frame = tk.Frame(self.root, bg="#f7f5f2")
        button_frame.place(relx=0.5, rely=0.9, anchor="center")

        if self.user_role == "Super Admin":
            self.btn_apply = Button(button_frame, text="APPLY PRIVACY ENHANCEMENT", font=("Helvetica", 14), borderwidth=2, relief="solid", command=self.start_background_capture, bg="#a83232", fg="#ffffff")
            self.btn_apply.grid(row=0, column=0, padx=10, pady=10)

            self.btn_store = Button(button_frame, text="STORE TO DATABASE", font=("Helvetica", 14), borderwidth=2, relief="solid", command=self.store_image, bg="#a83232", fg="#ffffff")
            self.btn_store.grid(row=0, column=1, padx=10, pady=10)

        self.btn_exit = Button(button_frame, text="EXIT", font=("Helvetica", 14), borderwidth=2, relief="solid", command=self.exit_app, bg="#a83232", fg="#ffffff")
        self.btn_exit.grid(row=0, column=2, padx=10, pady=10)

    def update_camera(self):
        ret, frame = self.capture.read()
        if ret:
            self.current_frame = frame
            frame_flipped = np.flip(frame, axis=1)
            frame_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)

            if self.user_role != "User":
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.before_canvas.create_image(0, 0, anchor='nw', image=imgtk)
                self.before_canvas.image = imgtk

            if self.privacy_mode and self.background is not None:
                enhanced_frame = self.apply_privacy_enhancement(frame_flipped)
                self.inpainting_done = True

                if self.user_role == "Super Admin":
                    frame_rgb_enhanced = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                    img_enhanced = Image.fromarray(frame_rgb_enhanced)
                    imgtk_enhanced = ImageTk.PhotoImage(image=img_enhanced)
                    self.after_canvas.create_image(0, 0, anchor='nw', image=imgtk_enhanced)
                    self.after_canvas.image = imgtk_enhanced

        self.root.after(10, self.update_camera)

    def start_background_capture(self):
        self.btn_apply.config(state=tk.DISABLED)
        self.root.update_idletasks()
        start_time = time.time()
        self.background = None
        while time.time() - start_time < 45:
            ret, background = self.capture.read()
            if ret:
                self.background = np.flip(background, axis=1)
        self.privacy_mode = True
        self.btn_apply.config(state=tk.NORMAL)
        messagebox.showinfo("Background Captured", "Background captured successfully. Privacy enhancement is now active.")

    def apply_privacy_enhancement(self, frame):
        if frame is not None:
            results = self.model(frame)[0]
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                mask = np.zeros_like(frame[:, :, 0])

                for i in range(len(boxes)):
                    if classes[i] == 0 and scores[i] > 0.5:
                        mask[masks[i] > 0.5] = 255

                inpainted_frame = frame.copy()
                inpainted_frame[mask == 255] = self.background[mask == 255]
                return inpainted_frame
        return frame

    def store_image(self):
        if self.current_frame is not None:
            image_name = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            cv2.imwrite(image_name, self.current_frame)
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
        self.root.quit()
        root = tk.Tk()
        database_name = "images.db"
        app = RoleSelection(root)
        root.mainloop()

class RoleSelection:
    def __init__(self, root):
        self.root = root
        self.root.title("Role Selection")
        self.root.geometry("500x300")

        self.selected_role = StringVar(self.root)
        self.selected_role.set("User")

        roles = ["User", "Admin", "Super Admin"]
        role_menu = OptionMenu(self.root, self.selected_role, *roles, command=self.on_role_selected)
        role_menu.pack(pady=20)

        self.login_frame = None

        self.proceed_button = Button(self.root, text="Proceed", command=self.proceed)
        self.proceed_button.pack(pady=10)

    def on_role_selected(self, event):
        role = self.selected_role.get()
        if role == "Super Admin":
            if self.login_frame is None:
                self.login_frame = tk.Frame(self.root)
                self.login_frame.pack(pady=10)

                self.username_label = Label(self.login_frame, text="Username")
                self.username_label.grid(row=0, column=0)
                self.username_entry = Entry(self.login_frame)
                self.username_entry.grid(row=0, column=1)

                self.password_label = Label(self.login_frame, text="Password")
                self.password_label.grid(row=1, column=0)
                self.password_entry = Entry(self.login_frame, show="*")
                self.password_entry.grid(row=1, column=1)
            else:
                self.login_frame.pack(pady=10)
        else:
            if self.login_frame:
                self.login_frame.pack_forget()

    def proceed(self):
        role = self.selected_role.get()
        if role == "Super Admin":
            username = self.username_entry.get()
            password = self.password_entry.get()
            if not self.verify_credentials(username, password):
                messagebox.showerror("Error", "Invalid Super Admin credentials!")
                return

        self.root.destroy()
        root = tk.Tk()
        database_name = "images.db"
        app = CameraApp(root, database_name, role)
        root.mainloop()

    def verify_credentials(self, username, password):
        conn = sqlite3.connect("credentials.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM super_admins WHERE username=? AND password=?", (username, password))
        result = cursor.fetchone()
        conn.close()
        return result is not None

if __name__ == "__main__":
    root = tk.Tk()
    role_selection = RoleSelection(root)
    root.mainloop()
