from Tkinter import *
import tkFileDialog
from PIL import ImageTk, Image
import detect
root = Tk(className="Image viewer")

canvas_width = 500
canvas_height = 500
root.config(bg="white")

def openimage():
    picfile = tkFileDialog.askopenfilename()
    print picfile
    if picfile:
        detect.scan(picfile)
        image = Image.open('detect.jpg')
        canvas.img = ImageTk.PhotoImage(image)
        canvas.create_image(0,0, anchor=NW, image=canvas.img) 
        canvas.configure(canvas, scrollregion=(0,0,canvas.img.width(),canvas.img.height()))


canvas = Canvas(root, width=canvas_width, height=canvas_height)
button = Button(root,text="Open",width=70,command=openimage)
status=Label(root,text = 'RED: Tay ga, GREEN: Xe so, BLUE: Xe dap',bg='gray',
            font=('Ubuntu',10),bd=2,fg='black',relief='sunken',anchor=W)
status.pack(side=BOTTOM,fill=X)
button.pack(side=BOTTOM)
canvas.pack(side=TOP)
mainloop()
