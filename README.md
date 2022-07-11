# Detection-des-plaques-d-immatriculation

Ici nous nous sommes propose

### 1. Nous avons telecharge les images de plaques d'immatriculation de vehicules sur Kaggle

### 2. Nous avons les avons labelise au format yolo (et obtenir le fichier .txt associe a chaque image) en utilant labelimg puis ziper et envoyer sur notre drive

### 3. Nous avons entrainee un modele de reseau de neurones a architecture Yolo v4 sur google colab en exploitant les gpu offert par Google 
 **-** On fait passer le jeu de donnee 1100 dans le reseau (1100 epoques)
 **-** L'entrainement a duree pres de 4h

### 4. Une fois le modele entraine, nous avons recuperer le fichier de configuration et de poids dont nous avons exploite pour faire des test en temps reel.

### 5. Le modele detecte bien les plaques d'immatriculation de vehicule

### 6. Une fois les plaques detectes, on les a fait passer dans un modele de detection de caracteres (tesseract) pour segmenter les differents caracteres issu de la plaque du vehicule.
