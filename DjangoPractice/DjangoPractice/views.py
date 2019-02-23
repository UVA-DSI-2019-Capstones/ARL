from django.contrib.auth.models import User
from django.http import Http404, JsonResponse
from gensim.models import LdaModel
import os
from shorttext.utils import standard_text_preprocessor_1
pre = standard_text_preprocessor_1()
from .models import TraineeResponseModel
from .serializers import UserSerializer, TraineeSerializer, MediaFileSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from gensim.corpora.dictionary import Dictionary
pre = standard_text_preprocessor_1()

from gensim.models import LdaModel
import os

dir = os.getcwd()
dir = os.path.dirname(dir)
dir = os.path.join(dir, 'db')

print('restarting views')
print('\n'* 5)
print(os.getcwd())


# Load LDA model
number_of_topics = 2
name = 'LDA_{}_topic'.format(number_of_topics)
temp_file = 'LDA_{}_topic.model'.format(number_of_topics)
temp_file = os.path.join(dir, 'LDA_models', temp_file)
LDA_2_topic  = LdaModel.load(temp_file)


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

      new_text = 'How many species of animals are there in Russia and in the US, and where are the biggest oceans?'
      tokens = [pre(new_text).split()]

      dict_test = Dictionary(tokens)
      bow_corpus_test = [dict_test.doc2bow(doc) for doc in tokens]

      LDA_2_topic.get_document_topics(bow=bow_corpus_test, minimum_probability=0.000001)

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