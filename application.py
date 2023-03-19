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
		file_menu.add_command(label='Recommencer', command=self.restart)
		file_menu.add_command(label='Tutoriel', command=self.tuto_window)
		file_menu.add_command(label='Quitter', command=self.destroy)
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
		w_frame.grid(row=1,column=1, columnspan=4)
		lbl = WrappingLabel(w_frame, text="Bienvenue sur l'application de projet FBI.\n Ce projet a pour but d'aider la police scientifique et les victimes d'aggressions à réaliser un portrait robot de l'agresseur grâce a l'intelligence artificielle.\nNous allons vous proposez 10 photos, choississez celles (maximum de 5) ressemblant à votre agresseur.\nLes photos sélectionnées seront encadrées en violet.\n Lorsque vous avez terminé votre sélection, veuillez cliquer sur le bouton (Sélection terminée).",font=self.font)
		lbl.pack(expand="True",fill=tk.X)

		tuto = Button(self, text="Voir le tutoriel",command=self.tuto_window)
		tuto.grid(row=2,column=3)

		lancer = Button(self, text="Lancer l'application",command=self.simulation)
		lancer.grid(row=2,column=2)
		

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
		self.consigne_lbl = WrappingLabel(self.consigne_frame, text="Veuillez choisir au moins une photo correspondant le plus à votre agresseur. Les images seclectionnées seront marquées par un contour indigo",font=self.font)
		self.consigne_lbl.pack(expand="True",fill=tk.X)
		self.selectedtxt_frame = Frame(self,bd=8)
		self.selectedtxt_frame.grid(row=0,column=0,sticky="nsew")
		self.selectedtxt_lbl = WrappingLabel(self.selectedtxt_frame, text="Voici les images que vous avez selectionné precedemment",font=self.font)
		self.selectedtxt_lbl.pack(expand="True",fill=tk.X)
		self.selected_frame = Frame(self,bd=8)
		self.selected_frame.grid(row=1,column=0,rowspan=4,sticky="nsew")
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

		self.bouton_flw = Button(self,command=self.ia)
		self.bouton_flw.grid(row=3,column=5,sticky="nsew")
		self.boutonlbl_flw = WrappingLabel(self.bouton_flw,text="Sélection terminée",font=self.font)
		self.boutonlbl_flw.pack(expand="True",fill=tk.X)
		self.frame_tour = Frame(self)
		self.frame_tour.grid(row=3,column=4,sticky="nsew")
		self.label_tour = WrappingLabel(self.frame_tour,text="Vous êtes au tour "+str(self.tour)+" sur 15",font=self.font)
		self.label_tour.pack(expand="True",fill=tk.X)


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
						self.label_tour = WrappingLabel(self.frame_tour,text="Vous êtes au tour "+str(self.tour)+" sur 15",font=self.font)
						self.label_tour.pack(expand="True",fill=tk.X)
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
				self.label_tour = WrappingLabel(self.frame_tour,text="Vous êtes au tour "+str(self.tour)+" sur 15",font=self.font)
				self.label_tour.pack(expand="True",fill=tk.X)
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
		self.top_level.title("Tutoriel")

		self.top_level.geometry(
		    "{}x{}+{}+{}".format(int(self.winfo_screenwidth()/2), int(self.winfo_screenheight()/2), int(0), int(0)))
		tuto_menubar = Menu(self.top_level)
		self.top_level.config(menu=tuto_menubar)
		file_tuto_menu=Menu(tuto_menubar, tearoff=False)
		file_tuto_menu.add_command(label='Quitter', command=self.top_level.destroy)
		tuto_menubar.add_cascade(
			label="Options",
			menu=file_tuto_menu,
			underline=0)

		# lbl = WrappingLabel(self.top_level,text="Tutoriel",font=self.font)
		# lbl.pack(expand="True",fill=tk.X)
		
		Grid.rowconfigure(self.top_level,0, weight=1)
		Grid.rowconfigure(self.top_level,1, weight=1)
		Grid.rowconfigure(self.top_level,2, weight=1)
		Grid.rowconfigure(self.top_level,3, weight=1)
		Grid.rowconfigure(self.top_level,4, weight=1)
		Grid.rowconfigure(self.top_level,5, weight=1)
		Grid.rowconfigure(self.top_level,6, weight=1)
		Grid.rowconfigure(self.top_level,7, weight=1)
		Grid.rowconfigure(self.top_level,8, weight=1)
		Grid.columnconfigure(self.top_level,0,weight=1)

		txt_introf = Frame(self.top_level)
		txt_introf.grid(row=0,column=0,sticky="nsew")
		txt_intro = WrappingLabel(txt_introf,text="Bienvenue dans ce tutoriel, nous allons vous guider dans l'utilisation de notre application.\nAprès avoir appuyer sur le bouton « Lancer l’application », plusieurs cas peuvent arriver :",font=self.font)
		txt_intro.pack(expand="True",fill=tk.X)
		txt_nonef = Frame(self.top_level)
		txt_nonef.grid(row=1,column=0,sticky="nsew")
		txt_none = WrappingLabel(txt_nonef,text="Aucune photo ne ressemble de près ou de loin à votre agresseur : vous pouvez soit choisir des portraits ayant un petit élément lui correspondant soit cliquer sur le bouton Recommencer se trouvant dans le menu déroulant Options.",font=self.font)
		txt_none.pack(expand="True",fill=tk.X)
		txt_onef = Frame(self.top_level)
		txt_onef.grid(row=2,column=0,sticky="nsew")
		txt_one = WrappingLabel(txt_onef,text="Seule une photo à des traits se rapprochant de votre agresseur mais ce n’est tout de même pas le portrait de ce dernier. Veuillez alors sélectionner l’image (en cliquant dessus) puis le bouton sélection terminée. Une fenêtre apparait alors : comme ce portrait ne correspond pas totalement à l’agresseur veuillez répondre non à la question posée. Vous passer au tour suivant.",font=self.font)
		txt_one.pack(expand="True",fill=tk.X)
		txt_lookf = Frame(self.top_level)
		txt_lookf.grid(row=3,column=0,sticky="nsew")
		txt_look = WrappingLabel(txt_lookf,text="Une photo correspond à votre agresseur. Veuillez alors sélectionner l’image (en cliquant dessus) puis le bouton sélection terminée. Une fenêtre apparait alors : comme ce portrait correspond à l’agresseur veuillez répondre oui à la question posée. C’est la fin de l’outil permettant d’émettre un portrait robot de l’agresseur. Une message d’information de fin s’affiche et l’image de l’agresseur peut être retrouver ainsi que toutes les images sélectionnées par la victime durant la création du portrait robot.",font=self.font)
		txt_look.pack(expand="True",fill=tk.X)
		txt_fewf = Frame(self.top_level)
		txt_fewf.grid(row=4,column=0,sticky="nsew")
		txt_few = WrappingLabel(txt_fewf,text="Différentes photos ont des caractéristiques se rapprochant du portrait de l’agresseur. Veuillez les sélectionner (avec un maximum de 5) et cliquer sur le bouton sélection terminée. Vous passez au tour suivant et vous pourrez observer toutes les photos sélectionnées depuis le début à gauche de votre écran.",font=self.font)
		txt_few.pack(expand="True",fill=tk.X)
		txt_endf = Frame(self.top_level)
		txt_endf.grid(row=5,column=0,sticky="nsew")
		txt_end = WrappingLabel(txt_endf,text="Si au bout de 15 tours vous n’avez pas trouver un portrait correspondant à l’agresseur, l’outil s’arrêtera automatique. Vous trouverez de la même façon les images sélectionnées dans le fichier « rapport_final.pdf »",font=self.font)
		txt_end.pack(expand="True",fill=tk.X)
		txt_backf = Frame(self.top_level)
		txt_backf.grid(row=6,column=0,sticky="nsew")
		txt_back = WrappingLabel(txt_backf,text="Si vous avez besoin de revenir sur ce tutoriel vous pouvez cliquer dans le menu en haut à gauche et sélectionner Tutoriel",font=self.font)
		txt_back.pack(expand="True",fill=tk.X)
		txt_quitf = Frame(self.top_level)
		txt_quitf.grid(row=7,column=0,sticky="nsew")
		txt_quit = WrappingLabel(txt_quitf,text="Si vous souhaitez quitter la page de Tutoriel vous pouvez cliquer sur le bouton « Fermer le tutoriel »",font=self.font)
		txt_quit.pack(expand="True",fill=tk.X)

		btn_quit=Button(self.top_level, text="Fermé le tutoriel", command=self.top_level.destroy)
		btn_quit.grid(row=8,column=0,sticky="e")


class ImButton(tk.Button):
    def __init__(self, parent, pilimage, image, *args, **kvargs):
    	self.pilimage = pilimage
    	self.image = image
    	super().__init__(parent, *args, image=self.pilimage, **kvargs)


class WrappingLabel(tk.Label):
    '''a type of Label that automatically adjusts the wrap to the size'''
    def __init__(self, master=None, **kwargs):
        tk.Label.__init__(self, master, **kwargs)
        self.bind('<Configure>', lambda e: self.config(wraplength=self.winfo_width()))



################
# Main program #
################

if __name__ == '__main__':
	app = Application()
	app.mainloop()