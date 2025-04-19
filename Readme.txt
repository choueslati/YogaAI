Les étapes a suivre pour lancer le projet
# activer l'env virtuel
.\env\Scripts\activate
# applique les migartions => creer la base de donnee
python manage.py migrate
##install les package requirements
pip install -r requirement.txt
# lancer le serveur
python manage.py runserver
