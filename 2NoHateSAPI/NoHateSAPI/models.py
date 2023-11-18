from django.db import models

# Create your models here.
class Info(models.Model):
    id = models.IntegerField(primary_key=True)
    content = models.CharField(max_length=999)