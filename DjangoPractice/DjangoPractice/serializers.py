from django.contrib.auth.models import User
from .models import TraineeResponseModel, MediaModel
from rest_framework import serializers

class UserSerializer(serializers.ModelSerializer):
  class Meta:
    model = User
    fields = ('id', 'username', 'first_name', 'last_name', 'email')


class TraineeSerializer(serializers.ModelSerializer):
  class Meta:
    model = TraineeResponseModel
    fields = ('id', 'avatar_prompt_id', 'identifier',
              'response_text', 'response_score', 'response_feedback', 'comment')

class MediaFileSerializer(serializers.ModelSerializer):
  class Meta():
    model = MediaModel
    fields = ('file', 'identifier')