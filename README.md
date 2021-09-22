# DNA_classification

this code provides my attempt to solve the dna classification problem. In response to the kaggle competition 
In this competition you will design deep learning algorithms to classify dna barcode sequences. 
Kaggle competition:

https://www.kaggle.com/t/9ced04b3b42f4d6aaaa88bf46bf60054



# Solutions 
1/ I first started with a simple dense neural net which had a not very good results . I used a simple encoding as following :
using this dic
dic= {'-':0,'A' :2**2 , 'C':2**3,'T':2**4,'G':2**5,'N':0}

by using a loop we get a encoding that is the same dim as the original len of the dna sequence , so I use a zero padding to get a uniform shape and all non DNA chars are turned to 0 as well _,N,etc.Using a custom dataset in pytorch I am able to then feed the data set to a dataloader as flat net 

I choose a 1296 so as to form a 36*36 square dim . I use this because I found the max len in the dna sequ is around 1058 .

after that I use a typical simple NN .


2/ usimg the same simple encodding in 1 I now use a cnn , I do reshaping to the output of the preprocessing to so as to be in the shape of 1,36,36 .1dconv .

both 1 and two are found under NN + CNN .ipynb.

3/2d CNN trained on word2vec embeddings of k-mers:

here I used gensim to generate the w2vec vecs using all the chars in the sequences . I then used Word2Vec(DNA_list , vector_size = 10 , min_count= 1 ) vector_size of 10 . I used iteration to generate the final DNA vectorized model which is feed to a dataloader .I do some croping to reduce the size of the DNA sequ to 600*10 after padding .I then reshape to get the 3 chennel image like shape of 3,40,50 .the same is done the test data without the training part . this is then inputed to a Conv . Here I use a pretrained model (structure only ) of the resnet18 , I do this to get a deep optimized CNN structure .


gensim is also updated to a newer version which might cause some issues so please update before starting the code using !pip install --update gensim 

file under word2vec_with_cnn_.ipynb

4/I also used a 2dConv nural net on a different type of one hot encodding 
here I use the following dic 
self.dic= {'-':[0,0,0,0],'A' :[1,0,0,0] , 'C':[0,1,0,0],'T':[0,0,1,0],'G':[0,0,0,1]}

I then multiply with the index of each char ,

using this and a resnet structure I got my best results in the competition 

file under transferLearning__kaggleComp_customPreTrain[0,0,0,1]bym.ipynb 


5/ I also used a LSTM (RNN) using the same encodding in (4)
class RNN(nn.Module):
    def __init__(self, input_size , hidden_size , num_layers , num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size , hidden_size , num_layers,batch_first= True)
        self.fc = nn.Linear(hidden_size*sequence_length ,num_classes)
        
        
    def forward(self,x):
        #print(x.shape) batchSize,72,72
        h0 = torch.zeros(self.num_layers , x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers , x.size(0) , self.hidden_size).to(device)
        #print(h0.shape) 2,15,275
        #print(x.shape)
        
        #Forward Prop 
        
        out,_ =  self.lstm(x,(h0,c0))
        #print(out.shape)15 , 72,256
        out = out.reshape(out.shape[0] ,-1)
        #print(out.shape) 15,18432
        out = self.fc(out)
        #print(out.shape) 15 ,1214
        return out 
        
which yielded a good result in the data of about 94% . I think I could do some hyperparameter tunning and get better results 

6/ Variational Autoencoding embedding with Neural Net 

as disscused in the lectures , I used a variational Autoencodder to produce a 50 dim representation of each DNA sequence , this is then feed to a dense nural net .


file under Variational AutoEncoder embedding with NN.ipynb


# how to run the code 

set the path by downloading the data folder 
train_features_path = '/train_features.csv'
test_features_path = '/test_features.csv'
train_labels_path = '/train_labels.csv'
test_labels_path = '/train_labels - Copy.csv'
to run the customataset you need to download the data folder and extract it , then for the feature data set please :
fulldataset = ( train_features.csv(path), train_labels.csv (path) ) 

testdataset = ( test_features.csv(path) , train_labels_copy.csv(path))





for all of the methods I use a softmax to determine the max probability of each seq and then determine on that the probability the sequence to be new dna .

