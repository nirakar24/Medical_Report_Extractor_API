from django.urls import path
from .views import CBCExtractView, ExtractorUI, signup_view, login_view, logout_view, whoami_view

urlpatterns = [
    path('extract/', CBCExtractView.as_view(), name='cbc-extract'),
    path('ui/', ExtractorUI.as_view(), name='extractor-ui'),
    path('auth/signup/', signup_view, name='signup'),
    path('auth/login/', login_view, name='login'),
    path('auth/logout/', logout_view, name='logout'),
    path('auth/whoami/', whoami_view, name='whoami'),
] 