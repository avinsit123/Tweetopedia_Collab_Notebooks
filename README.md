# Tweetopedia_Collab_Notebooks

This is a collection of all the notebooks that I used in order to train Neural Network Models for my web Project TweetoPedia.If you want to play around and contribute to the project by optimizing the neural network parameters go and follow the instructions given below.The Dependencies for this project are already given in the requirements.txt file in the Django Project.If you havent installed follow my instructions in the original Project to install it.
<br>


## 1.Query based Sentiment Analysis
Open <a href = "https://github.com/avinsit123/Tweetopedia_Collab_Notebooks/blob/master/Query_based_Convolutional_Sentiment_Analysis.ipynb" >this notebook</a> in collab by clicking the button on top of the notebook.
I have used a Convolutional Neural Network to perform sentiment Analysis.Go Ahead and Read the Notebook to understand more about how the process has been carried out.Run all the cells in a sequential order until you reach the cell with the contents
```python
BATCH_SIZE = 64
N_EPOCHS = 5
INPUT_DIM = len(TEXT.vocab)     #Do not change this
EMBEDDING_DIM = 100             #Embedding Dims for Glove vectors 
N_FILTERS = 100
FILTER_SIZES = [3,4,5]          #Size of Convolutional Filter
OUTPUT_DIM = 1                  #Output Dimensions = 1  
DROPOUT = 0.5
device = [ "cuda" if torch.cuda.is_available() else "cpu"]
model_name = "hateorlove"       #Name of check point file
```

These are the parameters for the Neural Network.Which you can change to improve Accuracy.Finally After you run all your cells.A checkpoint file will be generated.To Download the file,open the files tab on the right portion of the window.Double-Click on the .pth file and It would automatically start downloading the file.<br>

Copy the source path of the checkpoint file on your local computer.And open the python file with the below give directory.
```
/Tweet-o-pedia/mysite/QuerybasedSentiment/views.py
```
and move to the line
```python
def goloader(tweet):
   INPUT_DIM = 25002
   EMBEDDING_DIM = 100
   N_FILTERS = 100
   FILTER_SIZES = [3,4,5]
   OUTPUT_DIM = 1
   DROPOUT = 0.5 

   model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
   with open('/Users/r17935avinash/Downloads/hateorlove.pth', 'rb') as f:
       checkpoint = torch.load(f,map_location='cpu')
```
Replace the parameters that you changed in the above line of code and replace "/Users/r17935avinash/Downloads/hateorlove.pth" with your copied source path and run the project.

## 2.Trump Tweet Generator

In this project I have personally mined and cleaned 3240 Trump's Tweets.You can see my method <a href="https://github.com/avinsit123/Tweet_Like_Trump/tree/master/TweetlikeTrump">in this repository</a>.Now I have used a character-Level LSTM which takes in one character a time and predicts the next character.I have used Udacity's notebook Directly for this task.They have an amazing <a href="https://github.com/avinsit123/deep-learning-v2-pytorch">repository of code</a> containing pytorch implementaions of every NN Architecture possible.Open the <a href="https://github.com/avinsit123/Tweetopedia_Collab_Notebooks/blob/master/Trump_Tweet_Generator.ipynb">notebook</a> and start running all the cells sequentially until you reach the cell
```python
# define and print the net
n_hidden=512        #Dimensions of the hidden layer
n_layers=2          #No.of layers of LSTM
batch_size = 128    #Batch size(The smaller the better accurate)
seq_length = 100    #Sequence Length is the total no.of LSTMs in one layers side-by-side
n_epochs = 20       #Start smaller if you are just testing initial behavior
dropout = 0.5
lr1 = 0.001.         #This is the learning rate
```

These are the parameters for the Neural Network.Which you can change to improve Accuracy.Finally After you run all your cells.A checkpoint file will be generated.To Download the file,open the files tab on the right portion of the window.Double-Click on the .pth file and It would automatically start downloading the file.<br>

Copy the source path of the checkpoint file on your local computer.And open the python file with the below give directory.
```
/Tweet-o-pedia/mysite/Tweet_Generator/LSTMModel.py
```
and go to the line
```python
def LoadModel():
    with open('/Users/r17935avinash/Desktop/Trump_Tweet_Analysis/mysite/Tweet_Generator/rnn_final.net', 'rb') as f:
        checkpoint = torch.load(f,map_location='cpu')
```
Copy the Source Path of the .pth file and and replace '/Users/r17935avinash/Desktop/Trump_Tweet_Analysis/mysite/Tweet_Generator/rnn_final.net' with that path.

## Hate-o-meter

Open the <a href = "#">notebook</a>  by and open in collab using the button given at the top.Run all the cells sequentially until you reach the below given cell.
```python
model ,losses, validation_losses = train_model(target='is_hate_speech', hidden_dim=100,
                                        batch_size=50, epochs=5,
                                        print_every=150, log_every=5,
                                        num_layers=4, dropout=0.1)
```

Change the parameters of the above given cell to optimize the accuracy and continue running all the cells below.Finally,after running the final cell.Download checkpoint file and move to 
```
/Tweet-o-pedia/mysite/hate_o_meter/views.py
```
and then further to the line .
```python
with open('/Users/r17935avinash/Downloads/hateme.pth', 'rb') as f:
       checkpoint = torch.load(f,map_location='cpu')
```
and change the give path with the source path of the downloaded checkpoint file.
