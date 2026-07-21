# from django.contrib import admin
# from django.urls import include, path
# from django.conf.urls.static import static
# from django.conf import settings
# from core import views
# from django.views.generic.base import RedirectView

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('dashboard/', views.master_dashboard, name='master_dashboard'),
#     path('video_feed/', views.video_feed, name='video_feed'),  # Stream URL
#     path('', RedirectView.as_view(url='/dashboard/'), name='root'),
#     path('api/', include('core.urls')),
#     path('forensic/', views.forensic_dashboard, name='forensic_dashboard'),
#     path('api/search/', views.search_sightings, name='search_sightings'),
# ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static
from django.conf import settings
from core import views
from django.views.generic.base import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),

    # ✅ Landing FIRST
    path("", views.landing, name="landing"),

    path('dashboard/', views.master_dashboard, name='master_dashboard'),
    path('multicamera/', views.multicamera_dashboard, name='multicamera_dashboard'),
    path('forensic/', views.forensic_dashboard, name='forensic_dashboard'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path("api/today_stats/", views.today_stats, name="today_stats"),
    path('api/', include('core.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

