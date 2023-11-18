from rest_framework import viewsets
from . import models
from . import serializers
from pathlib import Path
from urllib import response

import json
from rest_framework.response import Response
from rest_framework.decorators import action

from .modelStruct.CNNforNLP import CNNForNLP
from .modelStruct.tokenizer import Tokenizer
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


class InfoViewset(viewsets.ModelViewSet):
    queryset = models.Info.objects.all()
    serializer_class = serializers.InfoSerializer



    #HERE = Path(__file__).parent
    #file = str(HERE) + '\modelo.pth'
    #model = CNNForNLP(31002,768,2,32,[3])
    #model.load_state_dict((torch.load(file,map_location ='cpu')))
    #model.eval()
    #print("modelo cargado")

    @action(detail=False,methods=['POST'])
    def use_model(self,request):

        HERE = Path(__file__).parent
        file = str(HERE) + '\modelo.pth'
        model = CNNForNLP(31002,768,2,32,[3])
        model.load_state_dict((torch.load(file,map_location ='cpu')))
        model.eval()
        #print("modelo cargado")

        #print("start using model")
        data = json.loads(request.body)
        
        print(data["text"])
        tokenizer = Tokenizer(data["text"])
        #print("tokenizer cargado")

        dataLoader = tokenizer.get()
        #print("data tokenized")

        with torch.no_grad():
            for batch in tqdm(dataLoader):
              b_input_ids = batch[0]#.to(device)
              b_input_mask = batch[1]#.to(device)
              b_labels = batch[2]#.to(device)

              predictions = model(b_input_ids,b_input_mask)
              pred = np.argmax(predictions,axis=1).flatten()

        print(pred)
        return Response(pred)


        