from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .chatbot import get_response

class ChatbotAPI(APIView):
    def post(self, request):
        user_message = request.data.get('message')
        if user_message:
            response = get_response(user_message)
            return Response({'response': response}, status=status.HTTP_200_OK)
        return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)
