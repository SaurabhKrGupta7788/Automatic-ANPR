from django.urls import path
from . import views

urlpatterns = [
    path('latest/', views.latest_sightings, name='latest_sightings'),
    path('search/', views.search_sightings, name='search_sightings'),  # ← Add this
    # ... other app paths if any
]