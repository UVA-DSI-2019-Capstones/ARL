from django.db import models

class TraineeResponseModel(models.Model):
  # id = models.AutoField()
  avatar_prompt_id = models.IntegerField()
  identifier = models.TextField()
  response_text = models.TextField()
  response_score = models.DecimalField(max_digits=4, decimal_places=2)
  response_feedback = models.TextField()
  comment = models.TextField()

  class Meta:
    db_table = 'trainee_response'


class MediaModel(models.Model):
  #This will upload the files in media
  file = models.FileField(blank=False, null=False, upload_to='uploads/')
  identifier = models.TextField(max_length=20)

  class Meta:
    db_table = 'media_table'