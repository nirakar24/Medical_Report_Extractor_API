from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .main import extract_report
from django.views.generic import TemplateView
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

# Create your views here.

class CBCExtractView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES.get('file')
        report_type = request.data.get('report_type', 'cbc').lower()
        if not file_obj:
            return Response({'error': 'No file uploaded.'}, status=status.HTTP_400_BAD_REQUEST)
        # Read file into memory
        file_bytes = file_obj.read()
        # Call the dispatcher
        result = extract_report(report_type, file_bytes)
        return Response(result)

class ExtractorUI(TemplateView):
    template_name = 'extractor/ui.html'

@csrf_exempt
@api_view(['POST'])
def signup_view(request):
    username = request.data.get('username')
    password = request.data.get('password')
    if not username or not password:
        return Response({'error': 'Username and password required.'}, status=400)
    if User.objects.filter(username=username).exists():
        return Response({'error': 'Username already exists.'}, status=400)
    user = User.objects.create_user(username=username, password=password)
    login(request, user)
    return Response({'success': True, 'username': user.username})

@csrf_exempt
@api_view(['POST'])
def login_view(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        return Response({'success': True, 'username': user.username})
    else:
        return Response({'error': 'Invalid credentials.'}, status=400)

@csrf_exempt
@api_view(['POST'])
def logout_view(request):
    logout(request)
    return Response({'success': True})

@api_view(['GET'])
def whoami_view(request):
    if request.user.is_authenticated:
        return Response({'authenticated': True, 'username': request.user.username})
    else:
        return Response({'authenticated': False})
