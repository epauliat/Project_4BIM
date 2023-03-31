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
from reportlab.pdfgen.canvas import Canvas
from datetime import date,datetime

# Internal librairies

from autoencoder_deployment import *
from geneticAlgo import *



#####################
# Class Application #
#####################

class Application(Tk):
	def __init__(self):
		"""Function that initializes the template for the window application
			Args: 
				None
			Returns: 
				None
		"""
		super().__init__()
		self.title("Application")
		self.geometry("{}x{}+{}+{}".format(self.winfo_screenwidth(), self.winfo_screenheight(), int(0), int(0)))
		self['background']='#CAD5CA'
		self.all_slt = []
		self.slt = []
		self.inc=0
		self.tour=1
		self.first=True
		self.font = ("Arial",10)
		for i in range(5):
			Grid.rowconfigure(self, i, weight=1)
			Grid.columnconfigure(self, i, weight=1)
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
			Returns:
				None
		"""
		self.destroy()
		self.__init__()


	def first_window(self):
		"""Function that creates the first window that shows up when the user start the program. It presents the project and asks one primilary question
			Args: 
				None
			Returns:
				None
		"""
		global w_frame, lbl, tuto, lancer, e
		w_frame = Frame(self,bg='#CAD5CA')
		w_frame.grid(row=1,column=1, columnspan=4)
		lbl = WrappingLabel(w_frame, text="Bienvenue sur l'application de projet FBI.\n Ce projet a pour but d'aider la police scientifique et les victimes d'aggressions à réaliser un portrait robot de l'agresseur grâce a l'intelligence artificielle.\nNous allons vous proposez 10 photos, choississez celles (maximum de 5) ressemblant à votre agresseur.\nLes photos sélectionnées seront encadrées en vert foncé.\n Lorsque vous avez terminé votre sélection, veuillez cliquer sur le bouton (Sélection terminée).",font=self.font,bg='#CAD5CA')
		lbl.pack(expand="True",fill=tk.X)

		tuto = Button(self, text="Voir le tutoriel", bg='#daebda',command=self.tuto_window)
		tuto.grid(row=2,column=3)

		lancer = Button(self, text="Lancer l'application", bg='#daebda', command=self.simulation)
		lancer.grid(row=2,column=2)
		

	def simulation(self):
		"""Function that creates the choices page where the victim will be presented a number of pictures from which they will have to choose the ones that match the best their attacker
			Args:
				None
			Returns:
				None
		"""
		w_frame.destroy()
		lbl.destroy()
		tuto.destroy()
		lancer.destroy()
		self.date = date.today().strftime("%Y_%m_%d")
		self.canva = Canvas("report_"+self.date+".pdf")
		self.canva.setFont("Times-Roman",12)
		self.consigne_frame = Frame(self,bg='#CAD5CA',bd=8,relief='raise')
		self.consigne_frame.grid(row=0,column=1,columnspan=5,sticky="nsew")
		self.consigne_lbl = WrappingLabel(self.consigne_frame, text="Veuillez choisir au moins une photo correspondant le plus à votre agresseur. Les images seclectionnées seront marquées par un contour vert foncé",font=self.font,bg='#CAD5CA')
		self.consigne_lbl.pack(expand="True",fill=tk.X)
		self.selectedtxt_frame = Frame(self,bd=8,bg='#CAD5CA')
		self.selectedtxt_frame.grid(row=0,column=0,sticky="nsew")
		self.selectedtxt_lbl = WrappingLabel(self.selectedtxt_frame, text="Voici les images que vous avez selectionné precedemment",font=self.font,bg='#CAD5CA')
		self.selectedtxt_lbl.pack(expand="True",fill=tk.X)
		self.selected_frame = Frame(self,bd=8,bg='#CAD5CA')
		self.selected_frame.grid(row=1,column=0,rowspan=4,sticky="nsew")
		self.bouton_flw = Button(self,command=self.ia,bg='#daebda')
		self.bouton_flw.grid(row=3,column=5,sticky="nsew")
		self.boutonlbl_flw = WrappingLabel(self.bouton_flw,text="Sélection terminée",font=self.font,bg='#daebda')
		self.boutonlbl_flw.pack(expand="True",fill=tk.X)
		self.frame_tour = Frame(self,bg='#CAD5CA')
		self.frame_tour.grid(row=3,column=4,sticky="nsew")
		self.new_images()


	def list_image(self):
		"""Function that gets the list of image that will be presented to the victim at each iteration
			Args:
				None
			Returns:
				array (self.list_img) : array of images to show
		"""
		self.list_img = []
		nb_faces = 999
		if self.first==True: 
			for i in range(10):
				x = random.randint(0, nb_faces)
				img = Image.open("faces/"+str(x)+".png")
				img.save("images/"+str(i)+".PNG")
				self.img=Image.open("images/"+str(i)+".PNG")
				self.resized_img = self.img.resize((190,190))
				self.list_img.append(self.resized_img)
			self.first=False
		else:
			ran = []
			for i in range(5):
				x = random.randint(0,9)
				while (x in ran):
					x = random.randint(0,9)
				print(x)
				ran.append(x)
				self.img = Image.open("images/"+str(x)+".PNG")
				self.resized_img = self.img.resize((190,190))
				self.list_img.append(self.resized_img)
			for i in range(5,10):
				x = random.randint(0, nb_faces)
				img = Image.open("faces/"+str(x)+".png")
				img.save("images/"+str(i)+".PNG")
				self.img=Image.open("images/"+str(i)+".PNG")
				self.resized_img = self.img.resize((190,190))
				self.list_img.append(self.resized_img)
		return self.list_img

	def selected(self,btn):
		"""Function that enables us to highlight the pictures that the user clicked on
			Args:
				btn (ImButton) : the button the user cliked on
			Returns:
				None
		"""
		if btn.cget('bg')!='#5a6650':
			btn.config(bg='#5a6650')
			self.slt.append(btn)
		else :
			btn.config(bg='#cad5ca')
			self.slt.remove(btn)

	def ia(self):
		"""Function that is called when we decide that we ended our selection, according to the number of pictures we choose it will either end the process, continue or will ask for another number of images choosen
			Args:
				None
			Returns:
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
						messagebox.showinfo("Message de fin","Vous avez trouvé un portrait correspondant à votre agresseur. Nous allons enregistrer ce portrait dans le dossier où vous vous trouvez.\nVous pourrez retrouver toutes les images sélectionnées dans le document (selected), celle de l'agresseur comprise.\n Nous avons de plus créer un rapport daté où vous pourrez observer l'image de l'agresseur.\nMerci pour votre collaboration.")
						picture_f = self.slt[0].image
						picture_f.save('agresseur.PNG')
						pic = Image.open('agresseur.PNG')
						self.canva.drawString(50,820,"Rapport de l'utilisation de l'intelligence artificielle dans le but de créer un portrait robot de l'agresseur.")
						self.canva.drawString(50,800,str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")+" (UTC+1)"))
						self.canva.drawString(50,760,"Image de l'agresseur:")
						self.canva.drawImage("agresseur.PNG",200,540)
						self.canva.drawString(20,20,"FBINSA 2022-2023. BAPT Zoé, BREANT Capucine, PAULIAT Eléa, ROUMANE Mathias, SCHULE Deborah")
						self.canva.save()
						self.destroy()
					else:
						self.not_found()
				else:
					messagebox.showerror("Nombre d'image selectionné incorrect","Veuillez choisir au minimum 1 portrait")
			elif len(self.slt)>5:
				messagebox.showerror("Nombre d'images selectionnés incorrect","Veuillez ne pas choisir plus de 5 portraits")
				self.slt = []
				for i in range(len(self.button)):
				 	self.button[i].configure(bg='#cad5ca')
			else:
				self.not_found()
		else:
			self.all_selected()
			messagebox.showinfo("Message de fin","Vous n'avez pas trouvé de portrait correspondant à votre agresseur dans le nombre d'itération autorisé. Les photos que vous aviez selectionnées sont enregistrées dans le dossier (selected). Merci pour votre collaboration.")
			self.destroy()


	def not_found(self):
		"""Function which computes new images from the ones selected (when it is not the agressor)
			Args:
				None
			Returns:
				None
		"""
		files = glob.glob('temp/*')
		for f in files:
			os.remove(f)
		for i in range(len(self.slt)):
			picture = self.slt[i].image
			picture.save('temp/'+str(i)+'.jpg')
		self.mutating(len(self.slt),1.5,"stds_of_all_encoded_vector_per_position.txt")
		self.all_selected()
		self.text = ScrolledText(self.selected_frame, wrap=WORD, width=30, height=45,bg='#CAD5CA')
		self.text.grid(row=2,column=0)
		self.text.images=[]
		for i in range(len(self.all_slt)):
			img = ImageTk.PhotoImage(self.all_slt[i].image.resize((160,160)))
			self.text.image_create(INSERT, padx=45, pady=5, image=img)
			self.text.images.append(img)
		self.slt = []
		self.tour+=1
		self.label_tour.destroy()
		self.new_images()

	def all_selected(self):
		"""Function that appends the list of selected images from the beginning and save them in a directory
			Args:
				None
			Returns:
				None
		"""
		for i in range(len(self.slt)):
			self.inc+=1
			self.all_slt.append(self.slt[i])
			picture = self.slt[i].image
			picture.save('selected/'+str(self.inc)+'.PNG')

	def new_images(self):
		"""Function that loads the images to be shown on the screen by assigning them to the ImButton
			Args:
				None
			Returns:
				None
		"""
		self.image = self.list_image()
		self.button = []
		for i in range(len(self.button)):
			self.button[i].destroy()
		for i in range(10):
			im_b = ImButton(self,ImageTk.PhotoImage(self.image[i]),self.image[i],bd=5,bg='#CAD5CA')
			self.button.append(im_b)
		for i in range(len(self.button)):
			self.button[i].configure(command=lambda button=self.button[i]: self.selected(button))
			if i<5:
				self.button[i].grid(row=1,column=i+1,sticky="nsew")
			else:
				j=i-5
				self.button[i].grid(row=2,column=j+1,sticky="nsew")
		self.label_tour = WrappingLabel(self.frame_tour,text="Vous êtes au tour "+str(self.tour)+" sur 15",font=self.font,bg='#CAD5CA')
		self.label_tour.pack(expand="True",fill=tk.X)


	def mutating(self,num, probability, std_file_path):
		"""Function that mutates the encoded images and prints them
			Args:
				num (int) : number of selected images
			    probability (float): probability used for mutations
			    std_file_path (str): path to the std txt file
			Returns:
			    None
		"""
		loaded_decoder=load_decoder("models/decoder.pt")
		loaded_encoder=load_encoder("models/encoder.pt")
		std = []
		with open(std_file_path) as f:
			std = f.readlines()
		stds = std[0].split(' ')
		for i in range(len(stds)):
			stds[i]=float(stds[i])
		vect_select=[]
		for i in range(num):
			vect_select.append(encoding_Image_to_Vector("temp/"+str(i)+".jpg",loaded_encoder))
		new_vectors=allNewvectors(vect_select,probability,stds)
		for i, vector in enumerate(new_vectors):
			decoded_pil=decoding_Vector_to_Image(vector,loaded_decoder)
			decoded_pil.save("images/"+str(i)+".PNG", format="png")


	def tuto_window(self):
		"""Function that creates a Toplevel windown to show a tutoriel of our application
			Args:
				None
			Returns:
				None
		"""
		self.top_level = tk.Toplevel()
		self.top_level.title("Tutoriel")
		self.top_level.geometry(
		    "{}x{}+{}+{}".format(int(self.winfo_screenwidth()/2), int(self.winfo_screenheight()/2), int(0), int(0)))
		self.top_level.configure(bg='#CAD5CA')
		tuto_menubar = Menu(self.top_level)
		self.top_level.config(menu=tuto_menubar)
		file_tuto_menu=Menu(tuto_menubar, tearoff=False)
		file_tuto_menu.add_command(label='Quitter', command=self.top_level.destroy)
		tuto_menubar.add_cascade(
			label="Options",
			menu=file_tuto_menu,
			underline=0)

		for i in range(9):
			Grid.rowconfigure(self.top_level,i,weight=1)
		Grid.columnconfigure(self.top_level,0,weight=1)

		txt_introf = Frame(self.top_level,bg='#CAD5CA')
		txt_introf.grid(row=0,column=0,sticky="nsew")
		txt_intro = WrappingLabel(txt_introf,text="Bienvenue dans ce tutoriel, nous allons vous guider dans l'utilisation de notre application.\nAprès avoir appuyer sur le bouton « Lancer l’application », plusieurs cas peuvent arriver :",font=self.font,bg='#CAD5CA')
		txt_intro.pack(expand="True",fill=tk.X)
		txt_nonef = Frame(self.top_level,bg='#CAD5CA')
		txt_nonef.grid(row=1,column=0,sticky="nsew")
		txt_none = WrappingLabel(txt_nonef,text="Aucune photo ne ressemble de près ou de loin à votre agresseur : vous pouvez soit choisir des portraits ayant un petit élément lui correspondant soit cliquer sur le bouton Recommencer se trouvant dans le menu déroulant Options.",font=self.font,bg='#CAD5CA')
		txt_none.pack(expand="True",fill=tk.X)
		txt_onef = Frame(self.top_level,bg='#CAD5CA')
		txt_onef.grid(row=2,column=0,sticky="nsew")
		txt_one = WrappingLabel(txt_onef,text="Seule une photo à des traits se rapprochant de votre agresseur mais ce n’est tout de même pas le portrait de ce dernier. Veuillez alors sélectionner l’image (en cliquant dessus) puis le bouton sélection terminée. Une fenêtre apparait alors : comme ce portrait ne correspond pas totalement à l’agresseur veuillez répondre non à la question posée. Vous passer au tour suivant.",font=self.font,bg='#CAD5CA')
		txt_one.pack(expand="True",fill=tk.X)
		txt_lookf = Frame(self.top_level,bg='#CAD5CA')
		txt_lookf.grid(row=3,column=0,sticky="nsew")
		txt_look = WrappingLabel(txt_lookf,text="Une photo correspond à votre agresseur. Veuillez alors sélectionner l’image (en cliquant dessus) puis le bouton sélection terminée. Une fenêtre apparait alors : comme ce portrait correspond à l’agresseur veuillez répondre oui à la question posée. C’est la fin de l’outil permettant d’émettre un portrait robot de l’agresseur. Une message d’information de fin s’affiche et l’image de l’agresseur peut être retrouver ainsi que toutes les images sélectionnées par la victime durant la création du portrait robot.",font=self.font,bg='#CAD5CA')
		txt_look.pack(expand="True",fill=tk.X)
		txt_fewf = Frame(self.top_level,bg='#CAD5CA')
		txt_fewf.grid(row=4,column=0,sticky="nsew")
		txt_few = WrappingLabel(txt_fewf,text="Différentes photos ont des caractéristiques se rapprochant du portrait de l’agresseur. Veuillez les sélectionner (avec un maximum de 5) et cliquer sur le bouton sélection terminée. Vous passez au tour suivant et vous pourrez observer toutes les photos sélectionnées depuis le début à gauche de votre écran.",font=self.font,bg='#CAD5CA')
		txt_few.pack(expand="True",fill=tk.X)
		txt_endf = Frame(self.top_level,bg='#CAD5CA')
		txt_endf.grid(row=5,column=0,sticky="nsew")
		txt_end = WrappingLabel(txt_endf,text="Si au bout de 15 tours vous n’avez pas trouver un portrait correspondant à l’agresseur, l’outil s’arrêtera automatique. Vous trouverez de la même façon les images sélectionnées dans le fichier « rapport_final.pdf »",font=self.font,bg='#CAD5CA')
		txt_end.pack(expand="True",fill=tk.X)
		txt_backf = Frame(self.top_level,bg='#CAD5CA')
		txt_backf.grid(row=6,column=0,sticky="nsew")
		txt_back = WrappingLabel(txt_backf,text="Si vous avez besoin de revenir sur ce tutoriel vous pouvez cliquer dans le menu en haut à gauche et sélectionner Tutoriel",font=self.font,bg='#CAD5CA')
		txt_back.pack(expand="True",fill=tk.X)
		txt_quitf = Frame(self.top_level,bg='#CAD5CA')
		txt_quitf.grid(row=7,column=0,sticky="nsew")
		txt_quit = WrappingLabel(txt_quitf,text="Si vous souhaitez quitter la page de Tutoriel vous pouvez cliquer sur le bouton « Fermer le tutoriel »",font=self.font,bg='#CAD5CA')
		txt_quit.pack(expand="True",fill=tk.X)
		btn_quit=Button(self.top_level, text="Fermé le tutoriel", command=self.top_level.destroy,bg='#daebda')
		btn_quit.grid(row=8,column=0,sticky="e")

##################
# Class ImButton #
##################

class ImButton(tk.Button):
    def __init__(self, parent, pilimage, image, *args, **kvargs):
    	"""Type of Button which has a Pillow Image and an Image as attribute
		    Args:
		    	None
		    Returns:
		    	None
	    """
    	self.pilimage = pilimage
    	self.image = image
    	super().__init__(parent, *args, image=self.pilimage, **kvargs)

#######################
# Class WrappingLabel #
#######################

class WrappingLabel(tk.Label):
    def __init__(self, master=None, **kwargs):
    	"""Type of Label that ajust to the window size
	    	Args:
	    		None
	    	Returns:
	    		None
    	"""
    	tk.Label.__init__(self, master, **kwargs)
    	self.bind('<Configure>', lambda e: self.config(wraplength=self.winfo_width()))


################
# Main program #
################

if __name__ == '__main__':
	app = Application()
	app.mainloop()