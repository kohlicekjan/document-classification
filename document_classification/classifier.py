import tkMessageBox
import ttk
from Tkinter import *


class GUI:
    def __init__(self, network):
        self.nt = network
        self.__render()

    def __render(self):
        self.window = Tk()
        self.window.geometry('400x400')
        self.window.title("Classifier document")

        self.button_classify = Button(self.window, text="Classify", command=self.classify_document)
        self.button_classify.config(height=2)
        self.button_classify.pack(expand=NO, fill=X, side=BOTTOM)

        self.progress = ttk.Progressbar(self.window, mode='indeterminate', maximum=40)

        self.text = Text()
        scroll = Scrollbar(self.window)
        scroll.config(command=self.text.yview)
        self.text.config(yscrollcommand=scroll.set)
        scroll.pack(side=RIGHT, fill=Y)
        self.text.pack(expand=YES, fill=BOTH)

        self.window.mainloop()

    def classify_document(self):

        text = self.text.get(1.0, END)

        if len(text) <= 10:
            tkMessageBox.showerror("ERROR", "Can not classify short text", parent=self.window)
            return

        self.__disabled(True)
        self.progress.start()

        label_name = self.nt.predict(text)

        self.progress.stop()
        self.progress.pack_forget()
        tkMessageBox.showinfo("Result", u"Category: {0}".format(label_name), parent=self.window)

        self.__disabled(False)

    def __disabled(self, disabled):
        if disabled:
            self.button_classify.config(state=DISABLED)
            self.text.config(state=DISABLED)
            self.progress.pack(expand=NO, fill=X, side=BOTTOM)
        else:
            self.button_classify.config(state=NORMAL)
            self.text.config(state=NORMAL)
            self.progress.pack_forget()
