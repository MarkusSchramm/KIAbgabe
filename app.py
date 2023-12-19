import pygame
import numpy as np
import seaborn as sns
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfile, askopenfile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from typing import Optional
import model as dm

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        pygame.mixer.init()
        self.data_model = dm.Model()
        self.trained = False
        self.pause = False
        self.musicfile = None
        self.instrument = 'Acoustic Grand Piano'
        self.notes = []
        self.geometry('1500x1200')
        self.title('KI Abgabe - Markus Schramm(11016480)')
        self.plot = False
        self.add_input()
        self.add_output()


    def add_input(self):
        self.menubar = tk.Menu(self, tearoff=0)
        self.file_menu = tk.Menu(self.menubar)
        self.config(menu=self.menubar)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Clear", command = self.clear_MIDI)
        self.file_menu.add_command(label="Exit", command=self.quit)

        #welcome text
        tk.Label(self, text = "Let's Make Some Music", font = ('bold', '12'), bg='white').grid(row = 0, column=0)
        self["bg"] = "white"

    def add_output(self):
        self.btnOpen = tk.Button(
            self, 
            text="Open MIDI", 
            bg="white", 
            fg="black",
            command=lambda: self.open_MIDI()
            )
        self.btnOpen.grid(row=1, column = 0)

        self.btnClear = tk.Button(
            self, 
            text="Clear MIDI", 
            bg="white", 
            fg="black",
            command=lambda: self.clear_MIDI()
            )
        self.btnClear.grid(row=1, column = 2)

        self.btnPlay = tk.Button(
            self, 
            text="Play MIDI", 
            bg="white", 
            fg="black",
            command=lambda: self.play_MIDI()
            )
        self.btnPlay.grid(row=2, column = 0)

        self.btnPause = tk.Button(
            self, 
            text="|| MIDI", 
            bg="white", 
            fg="black",
            command=lambda: self.pause_MIDI()
           )
        self.btnPause.grid(row=2, column = 1)

        self.btnStop = tk.Button(
            self, 
            text="Stop MIDI", 
            bg="white", 
            fg="black",
            command=lambda: self.stop_MIDI()
           )
        self.btnStop.grid(row=2, column = 2)
        self.btnTrain = tk.Button(
            self, 
            text="Train MIDI-Model", 
            bg="white", 
            fg="black",
            command=lambda: self.train_model()
           )
        self.btnTrain.grid(row=3, column = 0)
        self.btnGen = tk.Button(
            self, 
            text="Generate MIDI", 
            bg="white", 
            fg="black",
            command=lambda: self.generate_song()
           )
        self.btnGen.grid(row=3, column = 2)
    
    def plot_MIDI(self, row: Optional[int] = 0, col: Optional[int] = 0, count: Optional[int] = None):                
        fig = Figure(figsize=(5, 5))
        canvas = FigureCanvasTkAgg(fig, master=self)
        if len(self.notes) != 0: 
            if count:
                title = f'First {count} notes'
            else:
                title = f'Whole track'
                count = len(self.notes['pitch'])
            plot_pitch = np.stack([self.notes['pitch'], self.notes['pitch']], axis=0)
            plot_start_stop = np.stack([self.notes['start'], self.notes['end']], axis=0)
            fig.add_subplot(111).plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=col)

    def plot_distributions(self, row: Optional[int] = 0, col: Optional[int] = 0):
        fig = Figure(figsize=[15, 5])
        fig1 = fig
        fig2 = fig
        fig3 = fig

        
        fig1.add_subplot(131)
        sns.histplot(self.notes, x="pitch", bins=20)
        canvas1 = FigureCanvasTkAgg(fig1, master=self)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=row, column=0)

        fig2.add_subplot(131)
        max_step = np.percentile(self.notes['step'], 100 - 2.5)
        sns.histplot(self.notes, x="step", bins=np.linspace(0, max_step, 21))
        canvas2 = FigureCanvasTkAgg(fig1, master=self)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=row, column=1)


        fig3.add_subplot(131)
        max_duration = np.percentile(self.notes['duration'], 100 - 2.5)
        sns.histplot(self.notes, x="duration", bins=np.linspace(0, max_duration, 21))
        canvas3 = FigureCanvasTkAgg(fig1, master=self)
        canvas3.draw()
        canvas3.get_tk_widget().grid(row=row, column=2)
        

 
    def play_MIDI(self):
        if self.musicfile == None:
            self.open_MIDI()
        self.pause = False 
        pygame.mixer.music.load(self.musicfile)
        pygame.mixer.music.play()


    def stop_MIDI(self):
        self.pause = False
        pygame.mixer.music.stop()

    def pause_MIDI(self):
        if self.pause == False:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()
        self.pause = not self.pause

    def clear_MIDI(self):
        self.musicfile = None
        self.notes.clear()
    
    def open_MIDI(self):
        self.musicfile = askopenfilename()
        if self.musicfile == None: return
        self.notes = dm.Model.midi_to_notes(self.musicfile)
        self.plot_MIDI(5,4)
        #self.plot_distributions(5, 4)

    def train_model(self):
        self.trained = True
        self.data_model.train_model(70)

    def generate_song(self):
        if self.trained == True:
            self.notes = self.data_model.predict_notes()
            self.musicfile.write(self.notes)
        

def main():
    app = App()
    app.mainloop()

if __name__=='__main__':
    main()