###########
# Imports #
###########

# External librairies

import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import os
import glob
from Fusion import *

###########################
# Class Application #
###########################

class Application(Tk):
	def __init__(self):
		"""Function that initializes the template for the window application
		Args: 
			None
		Return: 
			None
		"""
		super().__init__()
		self.title("Application")
		self.title("Welcome")
		self.geometry("{}x{}+{}+{}".format(self.winfo_screenwidth(), self.winfo_screenheight(), int(0), int(0)))
		self.all_slt = []
		self.slt = []
		self.inc=0
		self.tour=1
		self.first=True
		self.font = ("Arial",10)
		Grid.rowconfigure(self, 0, weight=1)
		Grid.columnconfigure(self, 0, weight=1)
		Grid.rowconfigure(self, 1, weight=1)
		Grid.columnconfigure(self, 1, weight=1)
		Grid.rowconfigure(self, 2, weight=1)
		Grid.columnconfigure(self, 2, weight=1)
		Grid.rowconfigure(self, 3, weight=1)
		Grid.columnconfigure(self, 3, weight=1)
		Grid.rowconfigure(self, 4, weight=1)
		Grid.columnconfigure(self, 4, weight=1)
		Grid.columnconfigure(self, 5, weight=1)
		menubar = Menu(self)
		self.config(menu=menubar)
		file_menu=Menu(menubar, tearoff=False)
		file_menu.add_command(label='Restart', command=self.restart)
		file_menu.add_command(label='Tutoriel', command=self.tuto_window)
		file_menu.add_command(label='Exit', command=self.destroy)
		menubar.add_cascade(
			label="Options",
			menu=file_menu,
			underline=0)
		files = glob.glob('selected/*')
		for f in files:
			os.remove(f)
		self.first_window()	

	def restart(self):
		"""Function that allows us to restart our application
		Args:
			None
		Return:
			None
		"""
		self.destroy()
		self.__init__()


	def first_window(self):
		"""Function that creates the first window that shows up when the user start the program. It presents the project and asks one primilary question
		Args: 
			None
		Return:
			None
		"""
		global w_frame, lbl, tuto, lancer, e
		w_frame = Frame(self)
		w_frame.place(x=self.winfo_screenwidth()/4,y=self.winfo_screenheight()/3)
		lbl = Label(w_frame, text="Bienvenue sur l'application de projet FBI.\n Ce projet a pour but d'aider la police scientifique et les victimes d'aggressions à réaliser un portrait robot de l'agresseur grâce a l'intelligence artificielle.\nNous allons vous proposez 10 photos, choississez celles (maximum de 5) ressemblant à votre agresseur.\nLes photos sélectionnées seront encadrées en violet.\n Lorsque vous avez terminé votre sélection, veuillez cliquer sur le bouton (Sélection terminée).",font=self.font)
		lbl.grid(row=0,column=0,columnspan=2)

		tuto = Button(w_frame, text="Voir le tutoriel",command=self.tuto_window)
		tuto.grid(row=2,column=1)

		lancer = Button(w_frame, text="Lancer l'application",command=self.simulation)
		lancer.grid(row=2,column=0)
		

	def simulation(self):
		"""Function that creates the choices page where the victim will be presented a number of pictures from which they will have to choose the ones that match the best their attacker
		Args:
			None
		Return:
			None
		"""
		w_frame.destroy()
		lbl.destroy()
		tuto.destroy()
		lancer.destroy()
		self.consigne_frame = Frame(self,bd=8,relief='raise')
		self.consigne_frame.grid(row=0,column=1,columnspan=5,sticky="nsew")
		self.consigne_lbl = Label(self.consigne_frame, text="Veuillez choisir au moins une photo correspondant le plus à votre agresseur. Les images seclectionnées seront marquées par un contour indigo",font=self.font)
		self.consigne_lbl.grid(row=0,column=0,sticky="nsew")
		self.selected_frame = Frame(self,bd=8)
		self.selected_frame.grid(row=0,column=0,rowspan=5,sticky="nsew")
		self.selected_lbl = Label(self.selected_frame, text="Voici les images que vous avez selectionné precedemment",font=self.font)
		self.selected_lbl.grid(row=1,column=0,sticky="nsew")
		self.image = self.list_image()
		self.button = []
		for i in range(10):
			im_b = ImButton(self,ImageTk.PhotoImage(self.image[i]),self.image[i],bd=5)
			self.button.append(im_b)
		for i in range(len(self.button)):
			self.button[i].configure(command=lambda button=self.button[i]: self.selected(button))
			if i<5:
				self.button[i].grid(row=1,column=i+1,sticky="nsew")
			else:
				j=i-5
				self.button[i].grid(row=2,column=j+1,sticky="nsew")

		self.bouton_flw = Button(self,text="Sélection terminée",font=self.font,command=self.ia)
		self.bouton_flw.grid(row=3,column=5,sticky="nsew")
		self.label_tour = Label(self,text="Vous êtes au tour "+str(self.tour)+" sur 15",font=self.font)
		self.label_tour.grid(row=3,column=4,sticky="nsew")


	def list_image(self):
		"""Function that get the list of image that will be presented to the victim at each iteration
		Args:
			None
		Return:
			list : self.list_img
		"""
		self.list_img = []
		if self.first==True: #premier tour donc on va chercher les images dans la banque 
			for i in range(10):
				x = random.randint(0, 10000) #pour choisir aléatoirement des images dans la banque d'image
				img = Image.open("images/"+str(i)+".jpg")
				img.save("images/"+str(i)+".PNG")
				self.img=Image.open("images/"+str(i)+".PNG")
				self.resized_img = self.img.resize((190,190))
				self.list_img.append(self.resized_img)
			self.first=False
		else:
			for i in range(10):
				self.img = Image.open("images/"+str(i)+".PNG") #fichier à changer si on n'enregistre pas les images décodées ici
				self.resized_img = self.img.resize((190,190))
				self.list_img.append(self.resized_img)
		return self.list_img

	def selected(self,btn):
		"""Function that enables us to highlight the pictures that the user clicked on
		Args:
			btn (Button) : the button the user cliked on
		Return:
			None
		"""
		if btn.cget('bg')!='#6E00FF':
			btn.config(bg='#6E00FF')
			self.slt.append(btn)
		else :
			btn.config(bg='white')
			self.slt.remove(btn)

	def ia(self):
		"""Function that is called when we decide that we ended our selection, according to the number of pictures we choose it will either end the process, continue or will ask for another number of images choosen
		Args:
			None
		Return:
			None
		"""
		if self.tour<15:
			if len(self.slt)<2:
				if len(self.slt)==1:
					rep = messagebox.askquestion("Nombre d'image selectionné incorrect","Ce portrait correspond il à l'agresseur?")
					if rep == 'yes':
						self.inc+=1
						picture = self.slt[0].image
						picture.save('selected/'+str(self.inc)+'.PNG')
						messagebox.showinfo("Message de fin","Vous avez trouvé un portrait correspondant à votre agresseur. Nous allons enregistrer ce portrait dans le dossier où vous vous trouvez. Puis le portrait vous sera affiché. \nVous pourrez retrouver toutes les images sélectionnées dans le document (selected), celle de l'agresseur comprise.\nMerci pour votre collaboration.")
						picture_f = self.slt[0].image
						picture_f.save('agresseur.PNG')
						pic = Image.open('agresseur.PNG')
						pic.show()
						self.destroy()
					else:
						files = glob.glob('temp/*')
						for f in files:
							os.remove(f)
						for i in range(len(self.slt)):
							picture = self.slt[i].image
							picture.save('temp/'+str(i)+'.jpg')
						mutation(len(self.slt))
						self.inc+=1
						self.all_slt.append(self.slt[0])
						picture = self.slt[0].image
						picture.save('selected/'+str(self.inc)+'.PNG')
						self.text = ScrolledText(self.selected_frame, wrap=WORD, width=30, height=45)
						self.text.grid(row=2,column=0)
						self.text.images=[]
						for i in range(len(self.all_slt)):
							img = ImageTk.PhotoImage(self.all_slt[i].image.resize((160,160)))
							self.text.image_create(INSERT, padx=45, pady=5, image=img)
							self.text.images.append(img)
						self.slt = []
						self.image = self.list_image()
						self.button = []
						for i in range(len(self.button)):
							self.button[i].destroy()
						for i in range(10):
							im_b = ImButton(self,ImageTk.PhotoImage(self.image[i]),self.image[i],bd=5)
							self.button.append(im_b)
						for i in range(len(self.button)):
							self.button[i].configure(command=lambda button=self.button[i]: self.selected(button))
							if i<5:
								self.button[i].grid(row=1,column=i+1,sticky="nsew")
							else:
								j=i-5
								self.button[i].grid(row=2,column=j+1,sticky="nsew")
						self.tour+=1
						self.label_tour.destroy()
						self.label_tour = Label(self,text="Vous êtes au tour "+str(self.tour)+" sur 15",font=self.font)
						self.label_tour.grid(row=3,column=4,sticky="nsew")
				else:
					messagebox.showerror("Nombre d'image selectionné incorrect","Veuillez choisir au minimum 1 portrait")
			elif len(self.slt)>5:
				messagebox.showerror("Nombre d'images selectionnés incorrect","Veuillez ne pas choisir plus de 5 portraits")
				self.slt = []
				for i in range(len(self.button)):
				 	self.button[i].configure(bg='white')
			else:
				files = glob.glob('temp/*')
				for f in files:
					os.remove(f)
				for i in range(len(self.slt)):
					picture = self.slt[i].image
					picture.save('temp/'+str(i)+'.jpg')
				mutation(len(self.slt))
				for i in range(len(self.slt)):
					self.inc+=1
					self.all_slt.append(self.slt[i])
					picture = self.slt[i].image
					picture.save('selected/'+str(self.inc)+'.PNG')
				self.text = ScrolledText(self.selected_frame, wrap=WORD, width=30, height=45)
				self.text.grid(row=2,column=0)
				self.text.images=[]
				for i in range(len(self.all_slt)):
					img = ImageTk.PhotoImage(self.all_slt[i].image.resize((160,160)))
					self.text.image_create(INSERT, padx=45, pady=5, image=img)
					self.text.images.append(img)
				self.slt = []
				self.image = self.list_image()
				self.button = []
				for i in range(len(self.button)):
					self.button[i].destroy()
				for i in range(10):
					im_b = ImButton(self,ImageTk.PhotoImage(self.image[i]),self.image[i],bd=5)
					self.button.append(im_b)
				for i in range(len(self.button)):
					self.button[i].configure(command=lambda button=self.button[i]: self.selected(button))
					if i<5:
						self.button[i].grid(row=1,column=i+1,sticky="nsew")
					else:
						j=i-5
						self.button[i].grid(row=2,column=j+1,sticky="nsew")
				self.tour+=1
				self.label_tour.destroy()
				self.label_tour = Label(self,text="Vous êtes au tour "+str(self.tour)+" sur 15",font=self.font)
				self.label_tour.grid(row=3,column=4,sticky="nsew")
		else:
			for i in range(len(self.slt)):
				self.inc+=1
				self.all_slt.append(self.slt[i])
				picture = self.slt[i].image
				picture.save('selected/'+str(self.inc)+'.PNG')
			messagebox.showinfo("Message de fin","Vous n'avez pas trouvé de portrait correspondant à votre agresseur dans le nombre d'itération autorisé. Les photos que vous aviez selectionnées sont enregistrées dans le dossier (selected). Merci pour votre collaboration.")
			self.destroy()





	def tuto_window(self):
		"""Function that creates a Toplevel windown to show a tutoriel of our application
		Args:
			None
		Return:
			None
		"""
		self.top_level = tk.Toplevel()
		self.top_level.geometry(
		    "{}x{}+{}+{}".format(self.winfo_screenwidth(), self.winfo_screenheight(), int(0), int(0)))
		tuto_menubar = Menu(self.top_level)
		self.top_level.config(menu=tuto_menubar)
		file_tuto_menu=Menu(tuto_menubar, tearoff=False)
		file_tuto_menu.add_command(label='Exit', command=self.top_level.destroy)
		tuto_menubar.add_cascade(
			label="Options",
			menu=file_tuto_menu,
			underline=0)
		lbl = Label(self.top_level,text="Tutoriel",font=self.font).pack()
		btn2=Button(self.top_level, text="Close window", command=self.top_level.destroy).pack()

class ImButton(tk.Button):
    def __init__(self, parent, pilimage, image, *args, **kvargs):
    	self.pilimage = pilimage
    	self.image = image
    	super().__init__(parent, *args, image=self.pilimage, **kvargs)




################
# Main program #
################

if __name__ == '__main__':
	app = Application()
	app.mainloop()