"""
Graphical User Interface using Tkinter
Provides interactive interface for image retrieval with relevance feedback
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading

class ImageRetrievalGUI:
    def __init__(self, root, system):
        """
        Args:
            root: Tkinter root window
            system: ImageRetrievalSystem instance
        """
        self.root = root
        self.system = system
        
        # GUI state
        self.result_images = []
        self.result_indices = []
        self.result_scores = []
        self.relevant_set = set()
        self.irrelevant_set = set()
        self.feedback_iteration = 0
        
        # Setup GUI
        self.setup_window()
        self.create_widgets()
        
    def setup_window(self):
        self.root.title("Fashion-MNIST Image Retrieval System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="Fashion-MNIST Image Retrieval System",
            font=('Arial', 20, 'bold'),
            fg='white',
            bg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Search controls
        self.create_search_panel(main_container)
        
        # Right panel - Results
        self.create_results_panel(main_container)
        
        # Bottom panel - Feedback controls
        self.create_feedback_panel(main_container)
        
    def create_search_panel(self, parent):
        search_frame = tk.LabelFrame(
            parent, 
            text="Search Controls",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        search_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Text search
        tk.Label(
            search_frame, 
            text="Text Query:", 
            font=('Arial', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=(5, 2))
        
        self.text_entry = tk.Entry(search_frame, width=30, font=('Arial', 10))
        self.text_entry.pack(pady=(0, 5))
        self.text_entry.insert(0, "dress")
        
        tk.Button(
            search_frame,
            text="Search by Text",
            command=self.search_by_text,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=25,
            cursor='hand2'
        ).pack(pady=5)
        
        # Separator
        ttk.Separator(search_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Upload image
        tk.Label(
            search_frame,
            text="Upload Your Image:",
            font=('Arial', 10),
            bg='white'
        ).pack(anchor=tk.W, pady=(5, 2))
        
        tk.Label(
            search_frame,
            text="(Any size/color image)",
            font=('Arial', 8),
            fg='#7f8c8d',
            bg='white'
        ).pack(anchor=tk.W, pady=(0, 5))
        
        tk.Button(
            search_frame,
            text="Upload Image",
            command=self.upload_image,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=25,
            cursor='hand2'
        ).pack(pady=5)
        
        # Separator
        ttk.Separator(search_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Statistics
        tk.Label(
            search_frame,
            text="System Information:",
            font=('Arial', 10, 'bold'),
            bg='white'
        ).pack(anchor=tk.W, pady=(10, 5))
        
        self.stats_text = tk.Text(
            search_frame,
            height=8,
            width=30,
            font=('Arial', 9),
            bg='#ecf0f1',
            relief=tk.FLAT
        )
        self.stats_text.pack(pady=5)
        self.update_stats()
        
    def create_results_panel(self, parent):
        """Create results display panel"""
        results_frame = tk.LabelFrame(
            parent,
            text="Search Results (Top 20)",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbar
        canvas = tk.Canvas(results_frame, bg='white')
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=canvas.yview)
        
        self.results_container = tk.Frame(canvas, bg='white')
        self.results_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_container, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial message
        self.no_results_label = tk.Label(
            self.results_container,
            text="No results yet. Perform a search to see results.",
            font=('Arial', 12),
            fg='#7f8c8d',
            bg='white'
        )
        self.no_results_label.pack(pady=100)
        
    def create_feedback_panel(self, parent):
        """Create feedback control panel"""
        feedback_frame = tk.Frame(parent, bg='#f0f0f0')
        feedback_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Feedback info
        info_frame = tk.LabelFrame(
            feedback_frame,
            text="Relevance Feedback (Rocchio Algorithm)",
            font=('Arial', 11, 'bold'),
            bg='#2ecc71',
            fg='white',
            padx=10,
            pady=10
        )
        info_frame.pack(fill=tk.X)
        
        self.feedback_label = tk.Label(
            info_frame,
            text="Mark images as Relevant (✓) or Irrelevant (✗), then click Refine Search",
            font=('Arial', 10),
            bg='#2ecc71',
            fg='white'
        )
        self.feedback_label.pack(side=tk.LEFT, padx=10)
        
        # Feedback button
        tk.Button(
            info_frame,
            text="Refine",
            command=self.apply_feedback,
            bg='#27ae60',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            padx=20,
            pady=5
        ).pack(side=tk.RIGHT, padx=10)
        
        # Reset button
        tk.Button(
            info_frame,
            text="Reset Feedback",
            command=self.reset_feedback,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10),
            cursor='hand2'
        ).pack(side=tk.RIGHT, padx=5)
        
    def search_by_text(self):
        """Handle text search"""
        text_query = self.text_entry.get().strip()
        if not text_query:
            messagebox.showwarning("Warning", "Please enter a text query")
            return
        
        self.show_loading()
        
        def search_thread():
            try:
                indices, scores = self.system.search_by_text(text_query, top_k=20)
                self.root.after(0, lambda: self.display_results(indices, scores))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        threading.Thread(target=search_thread, daemon=True).start()
        
    def upload_image(self):
        """Handle image upload - accepts any size/color image"""
        filepath = filedialog.askopenfilename(
            title="Select Image (any size/color)",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Load image (can be color or grayscale, any size)
            img = Image.open(filepath)
            
            # Convert to grayscale if needed (Fashion-MNIST is grayscale)
            if img.mode != 'L':
                img = img.convert('L')
            
            # Resize to 28x28 (Fashion-MNIST size)
            img = img.resize((28, 28), Image.LANCZOS)
            img_array = np.array(img)
            
            self.show_loading()
            
            def search_thread():
                try:
                    indices, scores = self.system.search_by_uploaded_image(img_array, top_k=20)
                    self.root.after(0, lambda: self.display_results(indices, scores))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            
            threading.Thread(target=search_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}\n\nPlease select a valid image file.")
    
    def show_loading(self):
        """Show loading message"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        loading_label = tk.Label(
            self.results_container,
            text="Searching... Please wait",
            font=('Arial', 14),
            fg='#3498db',
            bg='white'
        )
        loading_label.pack(pady=100)
        self.root.update()
        
    def display_results(self, indices, scores):
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        self.result_indices = indices
        self.result_scores = scores
        self.result_images = []
        self.relevant_set.clear()
        self.irrelevant_set.clear()
        
        # Create grid of results (5 columns)
        for i, (idx, score) in enumerate(zip(indices, scores)):
            row = i // 5
            col = i % 5
            
            # Create frame for each result
            result_frame = tk.Frame(
                self.results_container,
                bg='white',
                relief=tk.RAISED,
                borderwidth=2
            )
            result_frame.grid(row=row, column=col, padx=5, pady=5)
            
            # Get and display image
            img = self.system.get_image(idx)
            img = img.resize((80, 80), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.result_images.append(photo)
            
            img_label = tk.Label(result_frame, image=photo, bg='white')
            img_label.pack()
            
            # Label and score
            label_text = f"{self.system.get_label(idx)}\n{score:.3f}"
            tk.Label(
                result_frame,
                text=label_text,
                font=('Arial', 8),
                bg='white'
            ).pack()
            
            # Feedback buttons
            btn_frame = tk.Frame(result_frame, bg='white')
            btn_frame.pack(pady=2)
            
            tk.Button(
                btn_frame,
                text="✓",
                command=lambda idx=idx: self.mark_relevant(idx),
                bg='#2ecc71',
                fg='white',
                font=('Arial', 10, 'bold'),
                width=3
            ).pack(side=tk.LEFT, padx=2)
            
            tk.Button(
                btn_frame,
                text="✗",
                command=lambda idx=idx: self.mark_irrelevant(idx),
                bg='#e74c3c',
                fg='white',
                font=('Arial', 10, 'bold'),
                width=3
            ).pack(side=tk.LEFT, padx=2)
        
        self.update_feedback_info()
        
    def mark_relevant(self, idx):
        """Mark image as relevant"""
        if idx in self.irrelevant_set:
            self.irrelevant_set.remove(idx)
        
        if idx in self.relevant_set:
            self.relevant_set.remove(idx)
        else:
            self.relevant_set.add(idx)
        
        self.update_feedback_info()
        
    def mark_irrelevant(self, idx):
        """Mark image as irrelevant"""
        if idx in self.relevant_set:
            self.relevant_set.remove(idx)
        
        if idx in self.irrelevant_set:
            self.irrelevant_set.remove(idx)
        else:
            self.irrelevant_set.add(idx)
        
        self.update_feedback_info()
        
    def apply_feedback(self):
        """Apply relevance feedback"""
        if not self.relevant_set and not self.irrelevant_set:
            messagebox.showinfo("Info", "Please mark at least one image as relevant or irrelevant")
            return
        
        self.show_loading()
        
        def feedback_thread():
            try:
                relevant_list = list(self.relevant_set)
                irrelevant_list = list(self.irrelevant_set)
                indices, scores = self.system.apply_relevance_feedback(
                    relevant_list, irrelevant_list, top_k=20
                )
                self.feedback_iteration += 1
                self.root.after(0, lambda: self.display_results(indices, scores))
                self.root.after(0, self.update_stats)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        threading.Thread(target=feedback_thread, daemon=True).start()
        
    def reset_feedback(self):
        """Reset feedback selections"""
        self.relevant_set.clear()
        self.irrelevant_set.clear()
        self.feedback_iteration = 0
        self.update_feedback_info()
        messagebox.showinfo("Info", "Feedback reset. Perform a new search.")
        
    def update_feedback_info(self):
        """Update feedback information display"""
        relevant_count = len(self.relevant_set)
        irrelevant_count = len(self.irrelevant_set)
        
        text = f"Relevant: {relevant_count} | Irrelevant: {irrelevant_count}"
        if self.feedback_iteration > 0:
            text += f" | Iteration: {self.feedback_iteration}"
        
        self.feedback_label.config(text=text)
        
    def update_stats(self):
        """Update system statistics"""
        session_stats = self.system.get_session_stats()
        if session_stats:
            stats_text += f"Session Info:\n"
            stats_text += f"  Iteration: {session_stats['iteration']}\n"
            stats_text += f"  Relevant: {session_stats['n_relevant']}\n"
            stats_text += f"  Irrelevant: {session_stats['n_irrelevant']}\n"
            stats_text += f"  Drift: {session_stats['query_drift']:.3f}\n"
        
        self.stats_text.insert('1.0', stats_text)
