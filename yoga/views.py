from django.shortcuts import render
from .models import Posture
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import base64
import json
import os
import mediapipe as mp
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from functools import wraps

# Définir le chemin du modèle
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model_3.h5') 

# Dictionnaire de correspondance entre les catégories de la base de données et les classes du modèle
CATEGORY_TO_CLASS = {
    'downdog': 'downdog',
    'goddess': 'goddess',
    'plank': 'plank',
    'tree': 'tree',
    'warrior2': 'warrior2'
}

# Classes reconnues par le modèle
CLASS_NAMES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

# Variable globale pour stocker le modèle
model = None

# Initialiser MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Fonction pour calculer l'angle entre 3 points
def calculate_angle(a, b, c):
    a = np.array([a['x'], a['y']])  # Convert to numpy arrays
    b = np.array([b['x'], b['y']])
    c = np.array([c['x'], c['y']])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

# Fonction pour dessiner les landmarks sur l'image
def draw_landmarks_on_image(image, pose_landmarks):
    # Obtenir les dimensions de l'image
    height, width, _ = image.shape
    
    # Dessiner les points
    for landmark in pose_landmarks:
        # Convertir les coordonnées normalisées en pixels
        x = int(landmark['x'] * width)
        y = int(landmark['y'] * height)
        
        # Dessiner le point si la visibilité est suffisante
        if landmark.get('visibility', 0) > 0.5:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    # Définir les connexions entre les points (similaires à POSE_CONNECTIONS de MediaPipe)
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Corps
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Jambes
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32)
    ]
    
    # Dessiner les connexions
    for connection in connections:
        start_idx, end_idx = connection
        
        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
            start_point = (int(pose_landmarks[start_idx]['x'] * width), 
                          int(pose_landmarks[start_idx]['y'] * height))
            end_point = (int(pose_landmarks[end_idx]['x'] * width), 
                        int(pose_landmarks[end_idx]['y'] * height))
            
            # Vérifier la visibilité des points
            if (pose_landmarks[start_idx].get('visibility', 0) > 0.5 and 
                pose_landmarks[end_idx].get('visibility', 0) > 0.5):
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    return image

def load_model_decorator(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        global model
        if model is None:
            try:
                model = load_model(MODEL_PATH, compile=False) 
                print("Modèle chargé avec succès (compile=False) par le décorateur")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle dans le décorateur: {e}")
                import traceback
                print(traceback.format_exc())
                return JsonResponse({'error': 'Erreur lors du chargement du modèle'}, status=500)
        return view_func(request, *args, **kwargs)
    return wrapper

def catalogue(request):
    categories = []
    cat_defs = [
        {'id': 1, 'name': 'Downdog'},
        {'id': 2, 'name': 'Goddess'},
        {'id': 3, 'name': 'Plank'},
        {'id': 4, 'name': 'Tree'},
        {'id': 5, 'name': 'Warrior'},
    ]

    for cat in cat_defs:
        posture = Posture.objects.filter(categorie=cat['name']).first()
        image_url = posture.image.url if posture else ''  # fallback si pas d'image
        categories.append({
            'id': cat['id'],
            'name': cat['name'],
            'image_url': image_url
        })

    return render(request, 'catalogue.html', {'categories': categories})

def category_detail(request, category_id):
    categories = {
        1: 'Downdog',
        2: 'Goddess',
        3: 'Plank',
        4: 'Tree',
        5: 'Warrior'
    }
    category_name = categories.get(category_id, 'Unknown')

    # Récupérer toutes les postures de la catégorie sélectionnée
    postures = Posture.objects.filter(categorie=category_name)

    return render(request, 'postures.html', {
        'category_name': category_name,
        'postures': postures
    })

def gallery(request):
    # Récupérer une posture aléatoire de chaque catégorie
    categories = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior']

    # Image principale (aléatoire parmi toutes les postures)
    main_posture = Posture.objects.order_by('?').first()

    # Récupérer des postures aléatoires pour chaque catégorie
    random_postures = []
    for category in categories:
        postures = Posture.objects.filter(categorie=category)
        if postures.exists():
            random_postures.append(random.choice(postures))

    # Si nous n'avons pas assez de postures, compléter avec des postures aléatoires
    while len(random_postures) < 4 and Posture.objects.count() > len(random_postures):
        posture = Posture.objects.order_by('?').first()
        if posture not in random_postures:
            random_postures.append(posture)

    return render(request, 'gallery.html', {
        'main_posture': main_posture,
        'random_postures': random_postures[:4]  # Limiter à 4 postures
    })

@csrf_exempt
@load_model_decorator
def detect_pose(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            landmarks_data = data.get('landmarks')  # Récupérer les landmarks envoyés par le client

            # Decode base64 image
            if not image_data:
                return JsonResponse({'error': 'No image data received'}, status=400)
            try:
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                np_arr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                return JsonResponse({'error': f'Error decoding image: {str(e)}'}, status=400)

            if img is None or img.size == 0:
                return JsonResponse({'error': 'Could not decode image data'}, status=400)

            # Vérifier si nous avons des landmarks
            pose_landmarks = None
            if landmarks_data and 'pose_landmarks' in landmarks_data:
                pose_landmarks = landmarks_data['pose_landmarks']
                
                # Dessiner les landmarks sur l'image avant de la passer au modèle
                img_with_landmarks = draw_landmarks_on_image(img.copy(), pose_landmarks)
                
                # Utiliser l'image avec landmarks pour la prédiction
                img_for_prediction = img_with_landmarks
            else:
                img_for_prediction = img

            # Preprocess image for model
            img_for_prediction = cv2.resize(img_for_prediction, (150, 150))
            img_for_prediction = img_for_prediction.astype('float32') / 255.0
            img_array = np.expand_dims(img_for_prediction, axis=0)

            # Make prediction
            if model is not None:
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions)
                confidence = float(predictions[0][predicted_class_idx])

                confidence_threshold = 0.4
                if confidence > confidence_threshold:
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    # Convertir le nom de classe du modèle en catégorie de la base de données
                    predicted_category = next((k for k, v in CATEGORY_TO_CLASS.items() if v == predicted_class), predicted_class)
                else:
                    predicted_class = ""
                    predicted_category = ""
            else:
                # Ceci ne devrait pas arriver si le décorateur fonctionne correctement
                predicted_class = ""
                predicted_category = ""
                confidence = 0.0

            # Calculer des angles si nous avons des landmarks
            angles = {}
            if pose_landmarks:
                # Exemple: calculer l'angle du coude droit
                try:
                    right_shoulder = pose_landmarks[11]  # Épaule droite
                    right_elbow = pose_landmarks[13]     # Coude droit
                    right_wrist = pose_landmarks[15]     # Poignet droit
                    
                    arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    angles['right_arm'] = arm_angle
                    
                    # Ajouter d'autres calculs d'angles selon les besoins
                except (IndexError, KeyError) as e:
                    print(f"Erreur lors du calcul des angles: {e}")

            # Retourner les landmarks avec la prédiction
            response_data = {
                'pose': predicted_class,
                'category': predicted_category,
                'confidence': confidence
            }
            
            # Inclure les landmarks et les angles dans la réponse
            if pose_landmarks:
                response_data['pose_landmarks'] = pose_landmarks
                response_data['angles'] = angles

            # Convertir l'image avec landmarks en base64 pour l'affichage
            if 'img_with_landmarks' in locals():
                _, buffer = cv2.imencode('.jpg', img_with_landmarks)
                img_with_landmarks_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data['image_with_landmarks'] = f"data:image/jpeg;base64,{img_with_landmarks_base64}"

            return JsonResponse(response_data)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
