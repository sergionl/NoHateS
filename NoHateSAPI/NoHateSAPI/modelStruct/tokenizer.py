from transformers import AutoTokenizer
from sklearn import preprocessing
import torch
from torch.utils.data import TensorDataset,DataLoader,RandomSampler
import pandas as pd
from pathlib import Path


class Tokenizer():
    def __init__(self,sentence):
        self.tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        le = preprocessing.LabelEncoder()

        HERE = Path(__file__).parent
        file = str(HERE) + '/test_labels.csv'
        #print(file)
        df = pd.read_csv(file,index_col=False)
        #print(df.shape)
        #print(df.head())
        test_labels = df.values.tolist()

        le.fit(test_labels)
        self.encoded_test_labels = le.transform(test_labels)
        #print(sentence)

        sentenceList = list()
        sentenceList.append(sentence)
        
        aux = list("0")
        encoded = le.transform(aux)
        #print(len(encoded))
        #print(len(sentenceList))
        #print(sentenceList)
        

        test_sent_index, test_input_ids, test_attention_masks, test_encoded_label_tensors = self.encoder_generator(sentenceList,encoded)
        test_dataset = TensorDataset(test_input_ids,test_attention_masks,test_encoded_label_tensors)

        bs=128

        self.test_data_loader = DataLoader(test_dataset,
                              sampler=RandomSampler(test_dataset),
                              batch_size=bs)

        pass

    def get(self):
        return self.test_data_loader

    def encoder_generator(self,sentences,labels):

        sent_index = []
        input_ids = []
        attention_masks =[]

        for index,sent in enumerate(sentences):

            sent_index.append(index)

            encoded_dict = self.tokenizer.encode_plus(sent,
                                                 add_special_tokens=True,
                                                 max_length=50,
                                                 padding='max_length',
                                                 truncation = True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
            input_ids.append(encoded_dict['input_ids'])

            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids,dim=0)#.cuda()
        attention_masks = torch.cat(attention_masks,dim=0)#.cuda()
        labels = torch.tensor(labels)#.cuda()
        sent_index = torch.tensor(sent_index)#.cuda()

        return sent_index,input_ids,attention_masks,labels