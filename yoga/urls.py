
from django.urls import path
from . import views

urlpatterns = [
    path('', views.catalogue, name='catalogue'),
    path('category/<int:category_id>/', views.category_detail, name='category_detail'),
    path('gallery/', views.gallery, name='gallery'),
    # dans ton_application/urls.py
    path('detect_pose/', views.detect_pose, name='detect_pose'),
    
]

