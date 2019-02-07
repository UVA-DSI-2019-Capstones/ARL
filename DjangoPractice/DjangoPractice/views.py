from django.contrib.auth.models import User
from django.http import Http404, JsonResponse

from .models import TraineeResponseModel
from .serializers import UserSerializer, TraineeSerializer, MediaFileSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

class TraineeList(APIView):
  """
  List all trainee response
  """
  def get(self, request, format=None):
    # db_name = connection.settings_dict['NAME']
    # print(db_name)

    trainee = TraineeResponseModel.objects.all()
    serializer = TraineeSerializer(trainee, many=True)
    return JsonResponse(serializer.data, safe=False)

class TraineeResponse(APIView):
  """
  Get the response of the trainee based on the identifier
  """
  def get_object(self, identifier):
    try:
      serializer = TraineeSerializer(TraineeResponseModel.objects.get(identifier=identifier))
      json_response = serializer.data
      json_response['success'] = True
      return json_response
    except TraineeResponseModel.DoesNotExist:
      #The identifier doesn't exist in the database
      failure_json = {'success': False}
      return failure_json



  def get(self, request, identifier, format=None):
    print(identifier)
    trainee = self.get_object(identifier)
    return JsonResponse(trainee, safe=False)

class UserList(APIView):


  """
  List all users, or create a new user.
  """


  def get(self, request, format=None):
    users = User.objects.all()
    serializer = UserSerializer(users, many=True)
    json_response = serializer.data
    json_response['success'] = True
    return Response(json_response)


  def post(self, request, format=None):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
      serializer.save()
      return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


  def delete(self, request, pk, format=None):
    user = self.get_object(pk)
    user.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


class UserDetail(APIView):
  """
  Retrieve, update or delete a user instance.
  """

  def get_object(self, pk):
    try:
      return User.objects.get(pk=pk)
    except User.DoesNotExist:
      raise Http404

  def get(self, request, pk, format=None):
    user = self.get_object(pk)
    user = UserSerializer(user)
    return Response(user.data)

  def put(self, request, pk, format=None):
    user = self.get_object(pk)
    serializer = UserSerializer(user, data=request.data)
    if serializer.is_valid():
      serializer.save()
      return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

  def delete(self, request, pk, format=None):
    user = self.get_object(pk)
    user.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


class MediaFileView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  def post(self, request, *args, **kwargs):
    file_serializer = MediaFileSerializer(data=request.data)
    files = request.FILES['file']


    print(files)
    if file_serializer.is_valid():
      print('Valid')

      file_serializer.save()
      return Response(file_serializer.data, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)