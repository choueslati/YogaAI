
from django.db import models
""" les tables a créer dans la base de données on les lance avec la migration """
class Posture(models.Model):
    CATEGORIES = [
        ('Downdog', 'Downdog'),
        ('Goddess', 'Goddess'),
        ('Plank', 'Plank'),
        ('Tree', 'Tree'),
        ('Warrior', 'Warrior'),
    ]
    nom = models.CharField(max_length=100)
    image = models.ImageField(upload_to='postures/')
    categorie = models.CharField(max_length=20, choices=CATEGORIES)

    def __str__(self):
        return self.nom
