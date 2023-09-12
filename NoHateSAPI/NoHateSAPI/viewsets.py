from rest_framework import viewsets
from . import models
from . import serializers
from pathlib import Path
from urllib import response

import json
from rest_framework.response import Response
from rest_framework.decorators import action

from .modelStruct.CNNforNLP import CNNForNLP
import torch
import torch.nn as nn


class InfoViewset(viewsets.ModelViewSet):
    queryset = models.Info.objects.all()
    serializer_class = serializers.InfoSerializer



    HERE = Path(__file__).parent
    file = str(HERE) + '\modelo.pth'
    model = CNNForNLP(31002,768,2,32,[3])
    model.load_state_dict((torch.load(file,map_location ='cpu')))
    model.eval()
    print("modelo cargado")

    @action(detail=False,methods=['POST'])
    def use_model(self,request):
        print("start using model")
        data = json.loads(request.body)
        
        print(data["text"])
        toTokenize = data["text"]

        
        return Response("c:")


        